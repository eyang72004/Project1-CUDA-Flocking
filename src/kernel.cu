#define GLM_FORCE_CUDA

#include <cuda.h>
#include "kernel.h"
#include "utilityCore.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <vector>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/device_vector.h>
// #include <thrust/unique.h>

#include <glm/glm.hpp>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

#define sharedBlockSize 32

#define GRID_LOOPING_OPTIMIZATION 1

#define SHARED_MEMORY_OPTIMIZATION 1

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;


// Attempt at extra credit: compact list containing only grid cells that currently contain boids
//int* dev_occupiedCellIndices;
//thrust::device_ptr<int> dev_thrust_occupiedCellIndices;

int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.
glm::vec3 *dev_pos_coherent;
glm::vec3 *dev_vel1_coherent;

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // Initialize velocity to 0
  cudaMemset(dev_vel1, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel1 failed!");

  cudaMemset(dev_vel2, 0, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMemset dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray << <fullBlocksPerGrid, blockSize >> >(1, numObjects,
    dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  //gridCellWidth = std::max(std::max(rule1Distance, rule2Distance), rule3Distance);

  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.

  // Part 2.1: allocate per-boid buffers used to construct uniform grid
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");


  //cudaMalloc((void**)&dev_occupiedCellIndices, N * sizeof(int));
  //checkCUDAErrorWithLine("cudaMalloc dev_occupiedCellIndices failed!");

  //dev_thrust_occupiedCellIndices = thrust::device_ptr<int>(dev_occupiedCellIndices);

  // We wrap the device arrays so Thrust can sort grid indices together with the corresponding particle-array indices.
  dev_thrust_particleArrayIndices = thrust::device_ptr<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices = thrust::device_ptr<int>(dev_particleGridIndices);

  // Each grid cell would store the range of sorted boids belonging to that cell.
  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // Part 2.3: temporary buffers for cell-coherent particle data
  cudaMalloc((void**)&dev_pos_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos_coherent failed!");

  cudaMalloc((void**)&dev_vel1_coherent, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1_coherent failed!");
  
  
  
  
  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids
  return glm::vec3(0.0f, 0.0f, 0.0f);
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1?
	int index = threadIdx.x + (blockIdx.x * blockDim.x);

    if (index >= N) {
        return;
    }

    const glm::vec3 selfPosition = pos[index];
	const glm::vec3 selfVelocity = vel1[index];

    // For Rule 1: we accumulate positions of nearby boids so we can steer towards their local center of mass.
    glm::vec3 perceivedCenter(0.0f);
	int rule1NeighborCount = 0;

	// For Rule 2: we accumulate a displacement away from boids that are too close. 
	glm::vec3 separation(0.0f);

	// For Rule 3: we accumulate the velocities of nearby boids so we can match their speed and align with neighbors.
	glm::vec3 perceivedVelocity(0.0f);
	int rule3NeighborCount = 0;

    // Naive neighbor search: every boid checks every other boid
    for (int i = 0; i < N; i++) {
        if (i == index) {
            continue;
        }

        glm::vec3 offset = pos[i] - selfPosition;
		float distance = glm::length(offset);

        if (distance < rule1Distance) {
			perceivedCenter += pos[i];
			++rule1NeighborCount;
        }

        if (distance < rule2Distance) {
            separation -= offset;
        }

        if (distance < rule3Distance) {
			perceivedVelocity += vel1[i];
			++rule3NeighborCount;
        }
    }

	glm::vec3 velocityChange(0.0f);

    // Cohesion
    if (rule1NeighborCount > 0) {
		perceivedCenter /= static_cast<float>(rule1NeighborCount);
		velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
    }

    // Separation
	velocityChange += separation * rule2Scale;

    // Alignment
    if (rule3NeighborCount > 0) {
		perceivedVelocity /= static_cast<float>(rule3NeighborCount);

        // The assignment pseudocode does not subtract selfVelocity it seems.
		velocityChange += perceivedVelocity * rule3Scale;
    }

	glm::vec3 newVelocity = selfVelocity + velocityChange;

	// Keep boid's speed bounded by maxSpeed
	float speed = glm::length(newVelocity);
	if (speed > maxSpeed) {
		newVelocity = (newVelocity / speed) * maxSpeed;
	}
	
    // vel1 is current timestep snapshot; vel2 stores the next timestep
    vel2[index] = newVelocity;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

// Attempt at Extra Credit: determine the grid-cell range that can contain boids within maximum interaction distance of this boid
__device__ void computeGridSearchBounds(
	glm::vec3 position,
	int gridResolution,
	glm::vec3 gridMin,
	float inverseCellWidth,
	int& minGridX,
	int& maxGridX,
	int& minGridY,
	int& maxGridY,
	int& minGridZ,
	int& maxGridZ
) {
	const float maxDistance = fmaxf(fmaxf(rule1Distance, rule2Distance), rule3Distance);

	glm::vec3 minPosition = position - glm::vec3(maxDistance);
	glm::vec3 maxPosition = position + glm::vec3(maxDistance);


	glm::vec3 minGridPosition = (minPosition - gridMin) * inverseCellWidth;

	glm::vec3 maxGridPosition = (maxPosition - gridMin) * inverseCellWidth;


	minGridX = imax(0, imin((int)floorf(minGridPosition.x), gridResolution - 1));


	maxGridX = imax(0, imin((int)floorf(maxGridPosition.x), gridResolution - 1));


	minGridY = imax(0, imin((int)floorf(minGridPosition.y), gridResolution - 1));
	maxGridY = imax(0, imin((int)floorf(maxGridPosition.y), gridResolution - 1));
	minGridZ = imax(0, imin((int)floorf(minGridPosition.z), gridResolution - 1));
	maxGridZ = imax(0, imin((int)floorf(maxGridPosition.z), gridResolution - 1));
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}

	// Convert the boid's world-space position into integer grid coordinates
	glm::vec3 gridPosition = (pos[index] - gridMin) * inverseCellWidth;

    int gridX = (int)floorf(gridPosition.x);
	int gridY = (int)floorf(gridPosition.y);
	int gridZ = (int)floorf(gridPosition.z);

    // Clamp coordinates so the resulting grid is always valid
	gridX = imax(0, imin(gridX, gridResolution - 1));
	gridY = imax(0, imin(gridY, gridResolution - 1));
	gridZ = imax(0, imin(gridZ, gridResolution - 1));

	// Keep original particle index parallel with its grid-cell index
	indices[index] = index;

	gridIndices[index] = gridIndex3Dto1D(gridX, gridY, gridZ, gridResolution);
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}
    
	int currentCell = particleGridIndices[index];

    // First particle in the sorted array begins its cell's range
    if (index == 0) {
        gridCellStartIndices[currentCell] = 0;
    } else {
		int previousCell = particleGridIndices[index - 1];


		if (currentCell != previousCell) {
            // We just entered a new cell
			gridCellStartIndices[currentCell] = index;

            // So the previous cell ended immediately before this index
			gridCellEndIndices[previousCell] = index;
		}
    }

    // Final particle closes the final occupied cell
	if (index == N - 1) {
		gridCellEndIndices[currentCell] = N;
	}
}

// Part 2.3: reorder particle data into grid-cell-coherent memory
__global__ void kernReorderData(
	int N, int *particleArrayIndices,
	glm::vec3* pos, glm::vec3 *vel1,
	glm::vec3* posCoherent, glm::vec3 *vel1Coherent
) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}


	int originalIndex = particleArrayIndices[index];

	// Reorder particle data to match the sorted grid-cell ordering
	posCoherent[index] = pos[originalIndex];
	vel1Coherent[index] = vel1[originalIndex];
}

__global__ void kernUpdateVelNeighborSearchScattered(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, float cellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2) {
	// TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
	// the number of boids that need to be checked.
	// - Identify the grid cell that this particle is in
	// - Identify which cells may contain neighbors. This isn't always 8.
	// - For each cell, read the start/end indices in the boid pointer array.
	// - Access each boid in the cell and compute velocity change from
	//   the boids rules, if this boid is within the neighborhood distance.
	// - Clamp the speed change before putting the new speed in vel2

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}

	const glm::vec3 selfPosition = pos[index];
	const glm::vec3 selfVelocity = vel1[index];

	// Determine this boid's grid coordinates
	glm::vec3 gridPosition = (selfPosition - gridMin) * inverseCellWidth;

	int gridX = (int)floorf(gridPosition.x);
	int gridY = (int)floorf(gridPosition.y);
	int gridZ = (int)floorf(gridPosition.z);

	gridX = imax(0, imin(gridX, gridResolution - 1));
	gridY = imax(0, imin(gridY, gridResolution - 1));
	gridZ = imax(0, imin(gridZ, gridResolution - 1));

	
	// With 2x-neighborhood-width cells, the neighborhood can overlap at most two cells along each dimension
	glm::vec3 cellCenter(
		gridMin.x + (gridX + 0.5f) * cellWidth,
		gridMin.y + (gridY + 0.5f) * cellWidth,
		gridMin.z + (gridZ + 0.5f) * cellWidth
	);

	int neighborX = gridX + ((selfPosition.x < cellCenter.x) ? -1 : 1);
	int neighborY = gridY + ((selfPosition.y < cellCenter.y) ? -1 : 1);
	int neighborZ = gridZ + ((selfPosition.z < cellCenter.z) ? -1 : 1);

	int xCells[2] = { gridX, neighborX };
	int yCells[2] = { gridY, neighborY };
	int zCells[2] = { gridZ, neighborZ };
	

	glm::vec3 perceivedCenter(0.0f);
	int rule1NeighborCount = 0;

	glm::vec3 separation(0.0f);
	glm::vec3 perceivedVelocity(0.0f);
	int rule3NeighborCount = 0;

	
	// Check only cells that the neighborhood can intersect
	for (int zIndex = 0; zIndex < 2; zIndex++) {
		int z = zCells[zIndex];

		if (z < 0 || z >= gridResolution) {
			continue;
		}

		for (int yIndex = 0; yIndex < 2; yIndex++) {
			int y = yCells[yIndex];

			if (y < 0 || y >= gridResolution) {
				continue;
			}

			for (int xIndex = 0; xIndex < 2; xIndex++) {
				int x = xCells[xIndex];

				

	// Part 2.2 - with cell width equal to the maximum neighborhood distance, check the current cell and one cell in each direction along every axis
	//for (int z = gridZ - 1; z <= gridZ + 1; z++) {

	//	if (z < 0 || z >= gridResolution) {
	//		continue;
	//	}

	//	for (int y = gridY - 1; y <= gridY + 1; y++) {
	//		if (y < 0 || y >= gridResolution) {
	//			continue;
	//		}

	//		for (int x = gridX - 1; x <= gridX + 1; x++) {


				if (x < 0 || x >= gridResolution) {
					continue;
				}

				int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

				int start = gridCellStartIndices[cellIndex];
				int end = gridCellEndIndices[cellIndex];


				// -1 marks empty cell
				if (start == -1 || end == -1) {
					continue;
				}

				// Cell range indexes particleArrayIndices,which then points to the boid's actual position and velocity data.
				for (int i = start; i < end; i++) {
					int neighborIndex = particleArrayIndices[i];

					if (neighborIndex == index) {
						continue;
					}

					glm::vec3 offset = pos[neighborIndex] - selfPosition;

					float distance = glm::length(offset);

					if (distance < rule1Distance) {
						perceivedCenter += pos[neighborIndex];
						++rule1NeighborCount;
					}

					if (distance < rule2Distance) {
						separation -= offset;
					}

					if (distance < rule3Distance) {
						perceivedVelocity += vel1[neighborIndex];
						++rule3NeighborCount;
					}
				}
			}
		}
	}

	glm::vec3 velocityChange(0.0f);

	// Rule 1 of Cohesion
	if (rule1NeighborCount > 0) {
		perceivedCenter /= static_cast<float>(rule1NeighborCount);
		velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
	}

	// Rule 2 of Separation
	velocityChange += separation * rule2Scale;

	// Rule 3 of Alignment
	if (rule3NeighborCount > 0) {
		perceivedVelocity /= static_cast<float>(rule3NeighborCount);
		velocityChange += perceivedVelocity * rule3Scale;
	}

	glm::vec3 newVelocity = selfVelocity + velocityChange;

	// Clamp speed to maxSpeed
	float speed = glm::length(newVelocity);

	if (speed > maxSpeed) {
		newVelocity = (newVelocity / speed) * maxSpeed;
	}

	vel2[index] = newVelocity;
}

// Attempt at extra credit: scattered uniform-grid search using dynamically computed min/max grid-cell bounds rather than a hard-coded 8- or 27-cell neighborhood
__global__ void kernUpdateVelNeighborSearchScatteredGridLoop(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, int* gridCellStartIndices,
	int* gridCellEndIndices, int* particleArrayIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2
) {

	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}

	const glm::vec3 selfPosition = pos[index];
	const glm::vec3 selfVelocity = vel1[index];

	// Compute the grid-cell bounds covering the maximum interaction distance around this boid
	int minGridX, maxGridX, minGridY, maxGridY, minGridZ, maxGridZ;


	computeGridSearchBounds(
		selfPosition, gridResolution, gridMin, inverseCellWidth,
		minGridX, maxGridX, minGridY, maxGridY, minGridZ, maxGridZ
	);

	glm::vec3 perceivedCenter(0.0f);
	int rule1NeighborCount = 0;


	glm::vec3 separation(0.0f);

	glm::vec3 perceivedVelocity(0.0f);
	int rule3NeighborCount = 0;

	// Traverse dynamically determined cell range.
	// x remains innermost because it is contiguous in gridIndex3Dto1D
	for (int z = minGridZ; z <= maxGridZ; z++) {
		for (int y = minGridY; y <= maxGridY; y++) {
			for (int x = minGridX; x <= maxGridX; x++) {


				int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

				int start = gridCellStartIndices[cellIndex];
				int end = gridCellEndIndices[cellIndex];

				// -1 marks empty cell
				if (start == -1 || end == -1) {
					continue;
				}

				// Scattered representation: the cell range indexes the particle-index array, which then points to boid data
				for (int i = start; i < end; i++) {
					int neighborIndex = particleArrayIndices[i];

					if (neighborIndex == index) {
						continue;
					}

					glm::vec3 offset = pos[neighborIndex] - selfPosition;

					float distance = glm::length(offset);

					if (distance < rule1Distance) {
						perceivedCenter += pos[neighborIndex];
						++rule1NeighborCount;
					}

					if (distance < rule2Distance) {
						separation -= offset;
					}

					if (distance < rule3Distance) {
						perceivedVelocity += vel1[neighborIndex];
						++rule3NeighborCount;
					}
				}
			}
		}
	}

	glm::vec3 velocityChange(0.0f);

	// Rule 1 of Cohesion
	if (rule1NeighborCount > 0) {
		perceivedCenter /= static_cast<float>(rule1NeighborCount);
		velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
	}

	// Rule 2 of Separation
	velocityChange += separation * rule2Scale;

	// Rule 3 of Alignment
	if (rule3NeighborCount > 0) {
		perceivedVelocity /= static_cast<float>(rule3NeighborCount);
		velocityChange += perceivedVelocity * rule3Scale;
	}

	glm::vec3 newVelocity = selfVelocity + velocityChange;

	float speed = glm::length(newVelocity);

	if (speed > maxSpeed) {
		newVelocity = (newVelocity / speed) * maxSpeed;
	}

	vel2[index] = newVelocity;
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2


	int index = threadIdx.x + (blockIdx.x * blockDim.x);

	if (index >= N) {
		return;
	}

	const glm::vec3 selfPosition = pos[index];
	const glm::vec3 selfVelocity = vel1[index];

	// Determine this boid's grid coordinates
	glm::vec3 gridPosition = (selfPosition - gridMin) * inverseCellWidth;

	int gridX = (int)floorf(gridPosition.x);
	int gridY = (int)floorf(gridPosition.y);
	int gridZ = (int)floorf(gridPosition.z);

	gridX = imax(0, imin(gridX, gridResolution - 1));
	gridY = imax(0, imin(gridY, gridResolution - 1));
	gridZ = imax(0, imin(gridZ, gridResolution - 1));


	// With cells twice the maximum neighborhood distance, the search region can overlap at most two cells along each dimension
	glm::vec3 cellCenter(
		gridMin.x + (gridX + 0.5f) * cellWidth,
		gridMin.y + (gridY + 0.5f) * cellWidth,
		gridMin.z + (gridZ + 0.5f) * cellWidth
	);

	int neighborX = gridX + ((selfPosition.x < cellCenter.x) ? -1 : 1);
	int neighborY = gridY + ((selfPosition.y < cellCenter.y) ? -1 : 1);
	int neighborZ = gridZ + ((selfPosition.z < cellCenter.z) ? -1 : 1);

	int xCells[2] = { gridX, neighborX };
	int yCells[2] = { gridY, neighborY };
	int zCells[2] = { gridZ, neighborZ };


	glm::vec3 perceivedCenter(0.0f);
	int rule1NeighborCount = 0;

	glm::vec3 separation(0.0f);

	glm::vec3 perceivedVelocity(0.0f);
	int rule3NeighborCount = 0;


	// x is the innermost loop since x is the contiguous dimension in gridIndex3Dto1D:
	// x + y * resolution + z * resolution^2
	for (int zIndex = 0; zIndex < 2; zIndex++) {
		int z = zCells[zIndex];

		if (z < 0 || z >= gridResolution) {
			continue;
		}

		for (int yIndex = 0; yIndex < 2; yIndex++) {
			int y= yCells[yIndex];
			if (y < 0 || y >= gridResolution) {
				continue;
			}
			for (int xIndex = 0; xIndex < 2; xIndex++) {
				int x = xCells[xIndex];
				if (x < 0 || x >= gridResolution) {
					continue;
				}
				int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
				int start = gridCellStartIndices[cellIndex];
				int end = gridCellEndIndices[cellIndex];
				// -1 marks empty cell
				if (start == -1 || end == -1) {
					continue;
				}

				// Part 2.3 difference: pos and vel1 have already been reordered so the cell range indexes them directly
				for (int i = start; i < end; i++) {
					if (i == index) {
						continue;
					}

					glm::vec3 offset = pos[i] - selfPosition;

					float distance = glm::length(offset);

					if (distance < rule1Distance) {
						perceivedCenter += pos[i];
						++rule1NeighborCount;
					}

					if (distance < rule2Distance) {
						separation -= offset;
					}

					if (distance < rule3Distance) {
						perceivedVelocity += vel1[i];
						++rule3NeighborCount;
					}
				}
			}

		}
	}

	glm::vec3 velocityChange(0.0f);

	// Cohesion
	if (rule1NeighborCount > 0) {
		perceivedCenter /= static_cast<float>(rule1NeighborCount);
		velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
	}

	// Separation
	velocityChange += separation * rule2Scale;

	// Alignment
	if (rule3NeighborCount > 0) {
		perceivedVelocity /= static_cast<float>(rule3NeighborCount);
		velocityChange += perceivedVelocity * rule3Scale;
	}

	glm::vec3 newVelocity = selfVelocity + velocityChange;

	float speed = glm::length(newVelocity);

	if (speed > maxSpeed) {
		newVelocity = (newVelocity / speed) * maxSpeed;
	}

	vel2[index] = newVelocity;
}

// Attempt at extra credit: coherent uniform-grid search using dynamically computed min/max grid-cell bounds rather than a hard-coded 8- or 27-cell neighborhood
__global__ void kernUpdateVelNeighborSearchCoherentGridLoop(
	int N, int gridResolution, glm::vec3 gridMin,
	float inverseCellWidth, int* gridCellStartIndices,
	int* gridCellEndIndices, glm::vec3* pos,
	glm::vec3* vel1, glm::vec3* vel2
) {
	int index = threadIdx.x + (blockIdx.x * blockDim.x);


	if (index >= N) {
		return;
	}


	const glm::vec3 selfPosition = pos[index];
	const glm::vec3 selfVelocity = vel1[index];

	// Compute the grid-cell bounds covering the maximum interaction distance around this boid
	int minGridX, maxGridX, minGridY, maxGridY, minGridZ, maxGridZ;


	computeGridSearchBounds(
		selfPosition, gridResolution, gridMin, inverseCellWidth,
		minGridX, maxGridX, minGridY, maxGridY, minGridZ, maxGridZ
	);

	glm::vec3 perceivedCenter(0.0f);
	int rule1NeighborCount = 0;


	glm::vec3 separation(0.0f);

	glm::vec3 perceivedVelocity(0.0f);
	int rule3NeighborCount = 0;

	// Traverse the dynamically determined cell range.
	// x remains innermost because it is contiguous in gridIndex3Dto1D

	for (int z = minGridZ; z <= maxGridZ; z++) {
		for (int y = minGridY; y <= maxGridY; y++) {
			for (int x = minGridX; x <= maxGridX; x++) {


				int cellIndex = gridIndex3Dto1D(x, y, z, gridResolution);

				int start = gridCellStartIndices[cellIndex];

				int end = gridCellEndIndices[cellIndex];


				// -1 marks empty cell
				if (start == -1 || end == -1) {
					continue;
				}


				// Coherent representation: the cell range indexes pos and vel1 directly
				for (int i = start; i < end; i++) {
					if (i == index) {
						continue;
					}

					glm::vec3 offset = pos[i] - selfPosition;

					float distance = glm::length(offset);

					if (distance < rule1Distance) {
						perceivedCenter += pos[i];
						++rule1NeighborCount;
					}


					if (distance < rule2Distance) {
						separation -= offset;
					}


					if (distance < rule3Distance) {
						perceivedVelocity += vel1[i];
						++rule3NeighborCount;
					}
				}
			}
		}
	}

	glm::vec3 velocityChange(0.0f);

	// Rule 1 of Cohesion
	if (rule1NeighborCount > 0) {
		perceivedCenter /= static_cast<float>(rule1NeighborCount);
		velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
	}

	// Rule 2 of Separation
	velocityChange += separation * rule2Scale;

	// Rule 3 of Alignment
	if (rule3NeighborCount > 0) {
		perceivedVelocity /= static_cast<float>(rule3NeighborCount);
		velocityChange += perceivedVelocity * rule3Scale;
	}

	glm::vec3 newVelocity = selfVelocity + velocityChange;

	float speed = glm::length(newVelocity);

	if (speed > maxSpeed) {
		newVelocity = (newVelocity / speed) * maxSpeed;
	}

	vel2[index] = newVelocity;
}

// Attempt at extra credit: use shared memory to reuse coherent neighbor-cell data among boids belonging to the same grid cell
__global__ void kernUpdateVelNeighborSearchCoherentShared(
	int N, int gridResolution, glm::vec3 gridMin, float inverseCellWidth,
	int* gridCellStartIndices, int* gridCellEndIndices,
	glm::vec3* pos, glm::vec3* vel1, glm::vec3* vel2
) {
	int selfCellIndex = blockIdx.x;

	if (selfCellIndex >= gridResolution * gridResolution * gridResolution) {
		return;
	}

	int selfStart = gridCellStartIndices[selfCellIndex];
	int selfEnd = gridCellEndIndices[selfCellIndex];


	// Empty grid cells have no boids for this block to update
	if (selfStart == -1 || selfEnd == -1) {
		return;
	}

	// Decode the block's 1D grid-cell index back into x, y, z
	//int gridX = selfCellIndex % gridResolution;
	//int gridY = (selfCellIndex / gridResolution) % gridResolution;
	//int gridZ = selfCellIndex / (gridResolution * gridResolution);

	// Neighbor data are loaded cooperatively into these shared-memory tiles
	__shared__ glm::vec3 sharedPos[sharedBlockSize];
	__shared__ glm::vec3 sharedVel[sharedBlockSize];

	// Union of the dynamic search bounds needed by active boids in this block chunk
	__shared__ int sharedMinGridX;
	__shared__ int sharedMaxGridX;
	__shared__ int sharedMinGridY;
	__shared__ int sharedMaxGridY;
	__shared__ int sharedMinGridZ;
	__shared__ int sharedMaxGridZ;


	// A grid cell may contain more boids than there are threads in one block, so we process the cell's own boids in block-sized chunks.
	for (int selfChunkStart = selfStart; selfChunkStart < selfEnd; selfChunkStart += blockDim.x) {

		int selfIndex = selfChunkStart + threadIdx.x;
		bool activeSelf = selfIndex < selfEnd && selfIndex < N;

		glm::vec3 selfPosition(0.0f);
		glm::vec3 selfVelocity(0.0f);


		int minGridX = 0;
		int maxGridX = -1;
		int minGridY = 0;
		int maxGridY = -1;
		int minGridZ = 0;
		int maxGridZ = -1;


		if (activeSelf) {
			selfPosition = pos[selfIndex];
			selfVelocity = vel1[selfIndex];


			// Compute the grid-cell bounds covering the maximum interaction distance around this boid
			computeGridSearchBounds(
				selfPosition, gridResolution, gridMin, inverseCellWidth,
				minGridX, maxGridX, minGridY, maxGridY, minGridZ, maxGridZ
			);


		}


		// Build the union of search bounds required by the active boids in this chunk
		if (threadIdx.x == 0) {
			sharedMinGridX = gridResolution;
			sharedMaxGridX = -1;
			sharedMinGridY = gridResolution;
			sharedMaxGridY = -1;
			sharedMinGridZ = gridResolution;
			sharedMaxGridZ = -1;
		}

		__syncthreads();

		if (activeSelf) {
			atomicMin(&sharedMinGridX, minGridX);
			atomicMax(&sharedMaxGridX, maxGridX);


			atomicMin(&sharedMinGridY, minGridY);
			atomicMax(&sharedMaxGridY, maxGridY);


			atomicMin(&sharedMinGridZ, minGridZ);
			atomicMax(&sharedMaxGridZ, maxGridZ);

		}

		__syncthreads();

		glm::vec3 perceivedCenter(0.0f);
		int rule1NeighborCount = 0;

		glm::vec3 separation(0.0f);

		glm::vec3 perceivedVelocity(0.0f);
		int rule3NeighborCount = 0;

		// Traverse only the union of cells actually required by the active boids
		// x remains innermost since it is contiguous in gridIndex3Dto1D
		for (int z = sharedMinGridZ; z <= sharedMaxGridZ; z++) {
			//if (z < 0 || z >= gridResolution) {
			//	continue;
			//}
			for (int y = sharedMinGridY; y <= sharedMaxGridY; y++) {
				//if (y < 0 || y >= gridResolution) {
				//	continue;
				//}
				for (int x = sharedMinGridX; x <= sharedMaxGridX; x++) {
					
					
					//if (x < 0 || x >= gridResolution) {
					//	continue;
					//}

					int neighborCellIndex = gridIndex3Dto1D(x, y, z, gridResolution);
					int neighborStart = gridCellStartIndices[neighborCellIndex];
					int neighborEnd = gridCellEndIndices[neighborCellIndex];


					// -1 marks empty cell
					if (neighborStart == -1 || neighborEnd == -1) {
						continue;
					}


					// Process arbitrarily large neighbor cells in tiles
					for (int neighborChunkStart = neighborStart; neighborChunkStart < neighborEnd; neighborChunkStart += blockDim.x) {
						
						int loadIndex = neighborChunkStart + threadIdx.x;

						int tileCount = imin((int)blockDim.x, neighborEnd - neighborChunkStart);

						// Cooperatively load one contiguous cell tile from global memory into shared memory
						if (threadIdx.x < tileCount) {
							sharedPos[threadIdx.x] = pos[loadIndex];
							sharedVel[threadIdx.x] = vel1[loadIndex];
						}


						__syncthreads();


						// Different boids in this block can have slightly different dynamic search bounds
						// We will only use this staged cell when it lies inside this boid's bounds
						bool cellInSearchBounds = activeSelf && x >= minGridX && x <= maxGridX && y >= minGridY && y <= maxGridY && z >= minGridZ && z <= maxGridZ;
						
						if (cellInSearchBounds) {
							for (int j = 0; j < tileCount; j++) {

								int neighborIndex = neighborChunkStart + j;

								if (neighborIndex == selfIndex) {
									continue;
								}

								glm::vec3 neighborPosition = sharedPos[j];

								glm::vec3 offset = neighborPosition - selfPosition;

								float distance = glm::length(offset);

								if (distance < rule1Distance) {
									perceivedCenter += neighborPosition;
									++rule1NeighborCount;
								}

								if (distance < rule2Distance) {
									separation -= offset;
								}

								if (distance < rule3Distance) {
									perceivedVelocity += sharedVel[j];
									++rule3NeighborCount;
								}
							}
						}

						// Every thread must finish using this tile before shared memory is reused for the next tile
						__syncthreads();
					}
				}
			}
		}

		if (activeSelf) {
			glm::vec3 velocityChange(0.0f);

			if (rule1NeighborCount > 0) {
				perceivedCenter /= static_cast<float>(rule1NeighborCount);
				velocityChange += (perceivedCenter - selfPosition) * rule1Scale;
			}

			velocityChange += separation * rule2Scale;

			if (rule3NeighborCount > 0) {
				perceivedVelocity /= static_cast<float>(rule3NeighborCount);
				velocityChange += perceivedVelocity * rule3Scale;
			}

			glm::vec3 newVelocity = selfVelocity + velocityChange;

			float speed = glm::length(newVelocity);

			if (speed > maxSpeed) {
				newVelocity = (newVelocity / speed) * maxSpeed;
			}

			vel2[selfIndex] = newVelocity;
		}

		// Ensure the whole block has finished this self-cell chunk before any thread advances to the next one
		__syncthreads();
	}
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
  // TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
  // TODO-1.2 ping-pong the velocity buffers

    dim3 fullBlocksPerGrid(
		(numObjects + blockSize - 1) / blockSize
	);

    // Compute the next velocity state from the current one
	kernUpdateVelocityBruteForce << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_pos,
		dev_vel1,
		dev_vel2
	);

	checkCUDAErrorWithLine("kernUpdateVelocityBruteForce failed!");
	

    // Newly computed velocities become the current velocities
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;
	

    // Advance positions using the new velocity state
    kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(
        numObjects,
        dt,
        dev_pos,
        dev_vel1
        );

	checkCUDAErrorWithLine("kernUpdatePos failed!");

	cudaDeviceSynchronize();
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

    dim3 fullBlocksPerGrid(
        (numObjects + blockSize - 1) / blockSize
    );

	dim3 gridBlocksPerGrid(
		(gridCellCount + blockSize - 1) / blockSize
	);

    // Label every boid with its original particle index and enclosing grid cell
    kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
        numObjects,
        gridSideCount,
        gridMinimum,
        gridInverseCellWidth,
        dev_pos,
        dev_particleArrayIndices,
        dev_particleGridIndices
        );

	checkCUDAErrorWithLine("kernComputeIndices failed!");


    // Sort by grid-cell index while carrying the corresponding particle index
    // Stable sort is not required by assignment, it seems
    thrust::sort_by_key(
		dev_thrust_particleGridIndices,
		dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices
    );

	checkCUDAErrorWithLine("thrust::sort_by_key failed!");

	

    // Mark every cell empty before filling ranges for occupied cells
	kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> >(
		gridCellCount,
		dev_gridCellStartIndices,
		-1
	);

	checkCUDAErrorWithLine("kernResetIntBuffer (start indices) failed!");

	kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> >(
        gridCellCount,
		dev_gridCellEndIndices,
		-1
	);

	checkCUDAErrorWithLine("kernResetIntBuffer (end indices) failed!");


	// Find contiguous sorted range belonging to each occupied grid cell
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_particleGridIndices,
		dev_gridCellStartIndices,
		dev_gridCellEndIndices
	);

	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// Update velocities using scattered uniform-grid neighbor search
	//kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(
	//	numObjects,
	//	gridSideCount,
	//	gridMinimum,
	//	gridInverseCellWidth,
	//	gridCellWidth,
	//	dev_gridCellStartIndices,
	//	dev_gridCellEndIndices,
	//	dev_particleArrayIndices,
	//	dev_pos,
	//	dev_vel1,
	//	dev_vel2
	//);

	// Cell start and end ranges now directly index dev_pos and dev_vel1
	//#if SHARED_MEMORY_OPTIMIZATION

	//kernUpdateVelNeighborSearchCoherentShared << <gridCellCount, blockSize >> > (
	//	numObjects,
	//	gridSideCount,
	//	dev_gridCellStartIndices,
	//	dev_gridCellEndIndices,
	//	dev_pos,
	//	dev_vel1,
	//	dev_vel2
	//);

	//checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherentShared failed!");



	#if GRID_LOOPING_OPTIMIZATION

		kernUpdateVelNeighborSearchScatteredGridLoop << <fullBlocksPerGrid, blockSize >> >(
			numObjects,
			gridSideCount,
			gridMinimum,
			gridInverseCellWidth,
			dev_gridCellStartIndices,
			dev_gridCellEndIndices,
			dev_particleArrayIndices,
			dev_pos,
			dev_vel1,
			dev_vel2
		);

		checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScatteredGridLoop failed!");

	#else
		kernUpdateVelNeighborSearchScattered << <fullBlocksPerGrid, blockSize >> >(
			numObjects,
			gridSideCount,
			gridMinimum,
			gridInverseCellWidth,
			gridCellWidth,
			dev_gridCellStartIndices,
			dev_gridCellEndIndices,
			dev_particleArrayIndices,
			dev_pos,
			dev_vel1,
			dev_vel2
		);

		checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	#endif

	// Newly computed velocities become the current velocity state
	glm::vec3* temp = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = temp;

	// Advance positions using the newly computed velocities
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dt,
		dev_pos,
		dev_vel1
	);

	checkCUDAErrorWithLine("kernUpdatePos failed!");

	cudaDeviceSynchronize();

}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.


	dim3 fullBlocksPerGrid(
		(numObjects + blockSize - 1) / blockSize
	);

	dim3 gridBlocksPerGrid(
		(gridCellCount + blockSize - 1) / blockSize
	);

	// Label each boid with its current array index and enclosing grid cell
	kernComputeIndices << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		gridSideCount,
		gridMinimum,
		gridInverseCellWidth,
		dev_pos,
		dev_particleArrayIndices,
		dev_particleGridIndices
	);

	checkCUDAErrorWithLine("kernComputeIndices failed!");


	// Sort particles by grid cell while preserving their original array indices
	thrust::sort_by_key(
		dev_thrust_particleGridIndices,
		dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices
	);

	checkCUDAErrorWithLine("thrust::sort_by_key failed!");

	//#if SHARED_MEMORY_OPTIMIZATION
	//	// Attempt at extra credit: compact the sorted grid-cell IDs into a list of occupied cells
	//	thrust::device_ptr<int> occupiedCellEnd = thrust::unique_copy(
	//		dev_thrust_particleGridIndices,
	//		dev_thrust_particleGridIndices + numObjects,
	//		dev_thrust_occupiedCellIndices
	//	);

	//	int occupiedCellCount =
	//		static_cast<int>(occupiedCellEnd - dev_thrust_occupiedCellIndices);

	//	checkCUDAErrorWithLine("thrust::unique_copy failed!");

	//#endif

	// Reset cell ranges before identifying occupied cells
	kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> >(
		gridCellCount,
		dev_gridCellStartIndices,
		-1
	);

	checkCUDAErrorWithLine("kernResetIntBuffer (start indices) failed!");

	kernResetIntBuffer << <gridBlocksPerGrid, blockSize >> >(
		gridCellCount,
		dev_gridCellEndIndices,
		-1
	);

	checkCUDAErrorWithLine("kernResetIntBuffer (end indices) failed!");

	// Determine contiguous sorted range belonging to each occupied cell
	kernIdentifyCellStartEnd << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_particleGridIndices,
		dev_gridCellStartIndices,
		dev_gridCellEndIndices
	);

	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");


	// Rearrange position and velocity data into the same cell-sorted order
	kernReorderData << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dev_particleArrayIndices,
		dev_pos,
		dev_vel1,
		dev_pos_coherent,
		dev_vel1_coherent
	);

	checkCUDAErrorWithLine("kernReorderData failed!");

	// Coherent buffers now become the active simulation state
	glm::vec3* tempPos = dev_pos;
	dev_pos = dev_pos_coherent;
	dev_pos_coherent = tempPos;

	glm::vec3* tempVel = dev_vel1;
	dev_vel1 = dev_vel1_coherent;
	dev_vel1_coherent = tempVel;


	// Cell start/end ranges now directly index dev_pos and dev_vel1
	#if SHARED_MEMORY_OPTIMIZATION

		kernUpdateVelNeighborSearchCoherentShared << <gridCellCount, sharedBlockSize >> >(
			numObjects,
			gridSideCount,
			gridMinimum,
			gridInverseCellWidth,
			dev_gridCellStartIndices,
			dev_gridCellEndIndices,
			dev_pos,
			dev_vel1,
			dev_vel2
		);

		checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherentShared failed!");

	

	#elif GRID_LOOPING_OPTIMIZATION
		kernUpdateVelNeighborSearchCoherentGridLoop << <fullBlocksPerGrid, blockSize >> >(
			numObjects,
			gridSideCount,
			gridMinimum,
			gridInverseCellWidth,
			dev_gridCellStartIndices,
			dev_gridCellEndIndices,
			dev_pos,
			dev_vel1,
			dev_vel2
		);

		checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherentGridLoop failed!");

	#else
	
		kernUpdateVelNeighborSearchCoherent << <fullBlocksPerGrid, blockSize >> >(
			numObjects,
			gridSideCount,
			gridMinimum,
			gridInverseCellWidth,
			gridCellWidth,
			dev_gridCellStartIndices,
			dev_gridCellEndIndices,
			dev_pos,
			dev_vel1,
			dev_vel2
		);

		checkCUDAErrorWithLine("kernUpdateVelNeighborSearchCoherent failed!");
	#endif

	// Newly computed velocities become the current velocity state
	tempVel = dev_vel1;
	dev_vel1 = dev_vel2;
	dev_vel2 = tempVel;


	// Advance reordered positions using new velocities
	kernUpdatePos << <fullBlocksPerGrid, blockSize >> >(
		numObjects,
		dt,
		dev_pos,
		dev_vel1
	);

	checkCUDAErrorWithLine("kernUpdatePos failed!");

	cudaDeviceSynchronize();
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
  //cudaFree(dev_occupiedCellIndices);

  cudaFree(dev_pos_coherent);
  cudaFree(dev_vel1_coherent);

  checkCUDAErrorWithLine("cudaFree buffers failed!");

}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");

  // Test for Part 2.1: verify grid indexing, sorting, and cell start/end ranges
  const int testN = 5;
  const int testGridResolution = 4;
  const int testGridCellCount = testGridResolution * testGridResolution * testGridResolution;


  const glm::vec3 testGridMin(0.0f);
  const float testCellWidth = 1.0f;
  const float testInverseCellWidth = 1.0f / testCellWidth;

  glm::vec3 testPositions[testN] = {
	  glm::vec3(0.25f, 0.25f, 0.25f), // cell 0
	  glm::vec3(1.25f, 0.25f, 0.25f), // cell 1
	  glm::vec3(1.75f, 0.25f, 0.25f), // cell 1
	  glm::vec3(0.25f, 1.25f, 0.25f), // cell 4
	  glm::vec3(0.25f, 0.25f, 1.25f)  // cell 16
  };

  glm::vec3* dev_testPositions;
  int* dev_testParticleIndices;
  int* dev_testGridIndices;
  int* dev_testCellStarts;
  int* dev_testCellEnds;


  cudaMalloc((void**)&dev_testPositions, testN * sizeof(glm::vec3));
  cudaMalloc((void**)&dev_testParticleIndices, testN * sizeof(int));
  cudaMalloc((void**)&dev_testGridIndices, testN * sizeof(int));
  cudaMalloc((void**)&dev_testCellStarts, testGridCellCount * sizeof(int));
  cudaMalloc((void**)&dev_testCellEnds, testGridCellCount * sizeof(int));

  checkCUDAErrorWithLine("2.1 Unit Test Allocation failed!");


  cudaMemcpy(
	  dev_testPositions,
	  testPositions,
	  testN * sizeof(glm::vec3),
	  cudaMemcpyHostToDevice
  );

  dim3 testBlocksPerGrid((testN + blockSize - 1) / blockSize);
  dim3 testCellBlocksPerGrid((testGridCellCount + blockSize - 1) / blockSize);


  kernComputeIndices << <testBlocksPerGrid, blockSize >> >(
	  testN,
	  testGridResolution,
	  testGridMin,
	  testInverseCellWidth,
	  dev_testPositions,
	  dev_testParticleIndices,
	  dev_testGridIndices
  );

  checkCUDAErrorWithLine("kernComputeIndices failed!");


  thrust::device_ptr<int> testThrustGridIndices(dev_testGridIndices);
  thrust::device_ptr<int> testThrustParticleIndices(dev_testParticleIndices);


  thrust::sort_by_key(
	  testThrustGridIndices,
	  testThrustGridIndices + testN,
	  testThrustParticleIndices
  );

  kernResetIntBuffer << <testCellBlocksPerGrid, blockSize >> >(
	  testGridCellCount,
	  dev_testCellStarts,
	  -1
  );

  kernResetIntBuffer << <testCellBlocksPerGrid, blockSize >> >(
	  testGridCellCount,
	  dev_testCellEnds,
	  -1
  );

  kernIdentifyCellStartEnd << <testBlocksPerGrid, blockSize >> >(
	  testN,
	  dev_testGridIndices,
	  dev_testCellStarts,
	  dev_testCellEnds
  );

  checkCUDAErrorWithLine("Cell range test failed!");
  cudaDeviceSynchronize();

  int testGridIndices[testN];
  int testParticleIndices[testN];
  int testCellStarts[testGridCellCount];
  int testCellEnds[testGridCellCount];


  cudaMemcpy(
	  testGridIndices,
	  dev_testGridIndices,
	  testN * sizeof(int),
	  cudaMemcpyDeviceToHost
  );

  cudaMemcpy(
	  testParticleIndices,
	  dev_testParticleIndices,
	  testN * sizeof(int),
	  cudaMemcpyDeviceToHost
  );


  cudaMemcpy(
	  testCellStarts,
	  dev_testCellStarts,
	  testGridCellCount * sizeof(int),
	  cudaMemcpyDeviceToHost
  );


  cudaMemcpy(
	  testCellEnds,
	  dev_testCellEnds,
	  testGridCellCount * sizeof(int),
	  cudaMemcpyDeviceToHost
  );

  std::cout << "Part 2.1 sorted grid indices: ";
  for (int i = 0; i < testN; i++) {
	  std::cout << testGridIndices[i] << " ";
  }
  std::cout << std::endl;


  std::cout << "cell 0 range: [" << testCellStarts[0] << ", " << testCellEnds[0] << ")" << std::endl;
  std::cout << "cell 1 range: [" << testCellStarts[1] << ", " << testCellEnds[1] << ")" << std::endl;
  std::cout << "cell 4 range: [" << testCellStarts[4] << ", " << testCellEnds[4] << ")" << std::endl;
  std::cout << "cell 16 range: [" << testCellStarts[16] << ", " << testCellEnds[16] << ")" << std::endl;


  cudaFree(dev_testPositions);
  cudaFree(dev_testParticleIndices);
  cudaFree(dev_testGridIndices);
  cudaFree(dev_testCellStarts);
  cudaFree(dev_testCellEnds);

  checkCUDAErrorWithLine("2.1 Unit Test cleanup failed!");

  // Test for 2.3: verify that particle data is reordered according to sorted particle-array indices
  const int reorderN = 4;

  glm::vec3 reorderPositions[reorderN] = {
	  glm::vec3(10.0f, 0.0f, 0.0f),
	  glm::vec3(20.0f, 0.0f, 0.0f),
	  glm::vec3(30.0f, 0.0f, 0.0f),
	  glm::vec3(40.0f, 0.0f, 0.0f)
  };

  glm::vec3 reorderVelocities[reorderN] = {
	  glm::vec3(1.0f, 0.0f, 0.0f),
	  glm::vec3(2.0f, 0.0f, 0.0f),
	  glm::vec3(3.0f, 0.0f, 0.0f),
	  glm::vec3(4.0f, 0.0f, 0.0f)
  };

  // Simulate a sorted particle-index array. 
  // Reordered data should come from particles 2, 0, 3, 1
  int reorderParticleIndices[reorderN] = { 2, 0, 3, 1 };

  glm::vec3* dev_reorderPositions;
  glm::vec3* dev_reorderVelocities;
  glm::vec3* dev_reorderPositionsCoherent;
  glm::vec3* dev_reorderVelocitiesCoherent;
  int* dev_reorderParticleIndices;


  cudaMalloc(
	  (void**)&dev_reorderPositions, reorderN * sizeof(glm::vec3)
  );

  cudaMalloc(
	  (void**)&dev_reorderVelocities, reorderN * sizeof(glm::vec3)
  );

  cudaMalloc(
	  (void**)&dev_reorderPositionsCoherent, reorderN * sizeof(glm::vec3)
  );

  cudaMalloc(
	  (void**)&dev_reorderVelocitiesCoherent, reorderN * sizeof(glm::vec3)
  );

  cudaMalloc(
	  (void**)&dev_reorderParticleIndices, reorderN * sizeof(int)
  );

  checkCUDAErrorWithLine("2.3 Unit Test Allocation failed!");


  cudaMemcpy(
	  dev_reorderPositions,
	  reorderPositions,
	  reorderN * sizeof(glm::vec3),
	  cudaMemcpyHostToDevice
  );

  cudaMemcpy(
	  dev_reorderVelocities,
	  reorderVelocities,
	  reorderN * sizeof(glm::vec3),
	  cudaMemcpyHostToDevice
  );

  cudaMemcpy(
	  dev_reorderParticleIndices,
	  reorderParticleIndices,
	  reorderN * sizeof(int),
	  cudaMemcpyHostToDevice
  );

  checkCUDAErrorWithLine("2.3 Unit Test memcpy failed!");

  dim3 reorderBlocksPerGrid((reorderN + blockSize - 1) / blockSize);

  kernReorderData << <reorderBlocksPerGrid, blockSize >> > (
	  reorderN,
	  dev_reorderParticleIndices,
	  dev_reorderPositions,
	  dev_reorderVelocities,
	  dev_reorderPositionsCoherent,
	  dev_reorderVelocitiesCoherent
  );

  checkCUDAErrorWithLine("kernReorderData failed!");

  cudaDeviceSynchronize();


  glm::vec3 reorderedPositions[reorderN];
  glm::vec3 reorderedVelocities[reorderN];

  cudaMemcpy(
	  reorderedPositions,
	  dev_reorderPositionsCoherent,
	  reorderN * sizeof(glm::vec3),
	  cudaMemcpyDeviceToHost
  );

  cudaMemcpy(
	  reorderedVelocities,
	  dev_reorderVelocitiesCoherent,
	  reorderN * sizeof(glm::vec3),
	  cudaMemcpyDeviceToHost
  );

  checkCUDAErrorWithLine("2.3 Unit Test memcpy back failed!");

  std::cout << "Part 2.3 reordered positions: ";
  for (int i = 0; i < reorderN; i++) {
	  std::cout << reorderedPositions[i].x << " ";
  }

  std::cout << std::endl;


  std::cout << "Part 2.3 reordered velocities: ";
  for (int i = 0; i < reorderN; i++) {
	  std::cout << reorderedVelocities[i].x << " ";
  }

  std::cout << std::endl;

  cudaFree(dev_reorderPositions);
  cudaFree(dev_reorderVelocities);
  cudaFree(dev_reorderPositionsCoherent);
  cudaFree(dev_reorderVelocitiesCoherent);
  cudaFree(dev_reorderParticleIndices);

  checkCUDAErrorWithLine("2.3 Unit Test cleanup failed!");

  // return;
}
