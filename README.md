**University of Pennsylvania, CIS 5650: GPU Programming and Architecture,
Project 1 - Flocking**

* Ethan Yang
  * [GitHub](https://github.com/eyang72004)
* Tested on: Windows 11 (Personal Laptop), Intel(R) Core(TM) Ultra 9 275HX, 32 GB DDR5 RAM, NVIDIA GeForce RTX 5060 Laptop GPU

> **Note:** All required implementation and performance results documented below were completed and collected before attempting the extra credit optimizations. I documented extra credit work separately at the end of this README.

## Boids Simulation

![20,000-boid scattered-grid simulation](images/boids_scattered_20000.gif)

*20,000-boid simulation using the scattered uniform-grid implementation with visualization enabled and `DT = 0.2`.*


![20,000-boid scattered-grid screenshot](images/boids_scattered_20000.png)


## Implementation

This project implements a CUDA flocking simulation using the three boid rules: cohesion, separation, alignment. I implemented the three required approaches:

1. **Naive:** Each boid checks every other boid when computing its velocity update.
2. **Scattered Uniform Grid:** Boids are assigned grid-cell indices and sorted by cell index. Cell start/end indices restrict the neighbor search to nearby grid cells, while the position and velocity arrays remain in their original order and the sorted particle-index array is used as an indirection. 
3. **Semi-Coherent Uniform Grid:** After sorting the grid indices and particle indices, I reordered the position and velocity data into grid-cell order. Neighbor searches can then directly access contiguous entries within the relevant cells rather than using the additional particle-index indirection.

For the required baseline grid implementation, the cell width is twice the maximum boid interaction radius. This allows the neighbor search to inspect at most 8 relevant cells in 3D. I separately benchmarked the alternative cell width equal to the interaction radius, which requires checking 27 cells.


# Performance Analysis

## Benchmark Methodology

I performed performance testing in **Release x64** mode on an **NVIDIA GeForce RTX 5060 Laptop GPU**, with **Vertical Sync disabled in the NVIDIA Control Panel**. The simulation timestep was `DT = 0.2`.

For the boid-count experiment, I tested these:
`1,000`, `2,500`, `5,000`, `10,000`, `20,000`, and `40,000` boids.


The default CUDA block size was 128 threads except during the block-size experiment. I measured performance using the application's existing wall-clock framerate meter. Both `VISUALIZE = 1` and `VISUALIZE = 0` were tested for the boid-count experiment.

For most configurations, I recorded five settled FPS readings and report their mean. I collected additional readings for a few noisy configurations. I also recorded some early visualization-on configurations as single settled readings, so I would not say that those values should be interpreted as having the same sampling depth as every repeated measurement.

These are application-level FPS measurements rather than isolated CUDA-kernel timings. In particular, even with `VISUALIZE = 0`, the current application still maps and unmaps its OpenGL buffers in `runCUDA()`. I did not use CUDA events for these measurements.


## 1. Effect of Increasing the Number of Boids

The measured framerates were these:

| Boids | Naive (Vis. On) | Scattered (Vis. On) | Coherent (Vis. On) | Naive (Vis. Off) | Scattered (Vis. Off) | Coherent (Vis. Off) |
|---:|---:|---:|---:|---:|---:|---:|
| 1,000 | 1240.30 | 1242.00 | 499.23 | 2630.82 | 2360.45 | 1308.48 |
| 2,500 | 981.20 | 1025.58 | 513.12 | 1607.54 | 1922.68 | 1225.16 |
| 5,000 | 761.80 | 929.28 | 538.08 | 1081.68 | 1759.70 | 1116.78 |
| 10,000 | 506.60 | 1045.50 | 503.46 | 621.36 | 1733.18 | 1031.66 |
| 20,000 | 261.10 | 466.96 | 504.52 | 293.90 | 917.74 | 1153.78 |
| 40,000 | 78.80 | 454.56 | 981.84 | 81.26 | 755.78 | 1672.40 |


### Visualization ON

![Framerate Vs. Boid Count with visualization on](images/framerate_vs_boids_visualization_on.png)

### Visualization OFF

![Framerate Vs. Boid Count with visualization off](images/framerate_vs_boids_visualization_off.png)


### Analysis

The **naive implementation shows the clearest degradation as the boid count increases**. With visualization disabled, it fell from 2630.82 FPS at 1000 boids to only 81.26 FPS at 40,000 boids. This behavior is consistent with the structure of the naive algorithm: each boid considers every other boid during the neighbor search, so the number of pairwise comparisons grows approximately quadratically with the number of boids.

The grid implementations avoid searching the complete boid population for every boid; however, that reduction is not free. Each grid-based frame also requires grid-index computation, sorting, cell-range construction, and, for the semi-coherent implementation, data reordering. At small populations, those preprocessing costs can outweigh the savings from reducing neighbor comparisons. This is evident at 1,000 boids, where the naive implementation is competitive with or faster than the grid approaches. As the boid count increases, however, the quadratic growth of the naive all-pairs search becomes dominant and the grid methods pull substantially ahead.

For instance, if we disable visualization at 40,000 boids, the naive implementation measured 81.26 FPS, compared with 755.78 FPS for the scattered grid and 1672.40 FPS for the coherent grid.

The grid measurements are not perfectly monotonic. Most notably, the coherent implementation increased substantially at 40,000 boids in both the visualization-on and visualization-off measurements. I have retained this measured result rather than smoothing or discarding it. Since these measurements use whole-application FPS rather than per-kernel CUDA event timings, I do not have sufficient evidence to attribute that increase to a specific GPU mechanism. I reckon that more fine-grained profiling would be necessary to determine its exact cause.

Visualization generally reduces the amount of simulation throughput because visualization-enabled frames additionally copy boid data to the VBO and perform rendering. The comparison should nevertheless be interpreted as application-level performance rather than pure rendering overhead, especially given the variability visible in several grid measurements.

## 2. Effect of CUDA Block Size and Block Count

For this experiment, I fixed the simulation at **20,000 boids**, used `VISUALIZE = 0`, and tested block sizes of 32, 64, 128, 256, and 512 threads.

Since the boid count remained fixed, changing the block size also changed the number of blocks launched according to:

`blockCount = ceil(N / blockSize)`

| Block Size (threads) | Block Count | Naive FPS | Scattered FPS | Coherent FPS |
|---:|---:|---:|---:|---:|
| 32 | 625 | 187.16 | 1428.70 | 1841.36 |
| 64 | 313 | 295.10 | 895.38 | 1704.66 |
| 128 | 157 | 293.90 | 917.74 | 1153.78 |
| 256 | 79 | 289.22 | 1614.36 | 1828.84 |
| 512 | 40 | 274.44 | 1565.98 | 1856.36 |


![Framerate vs. CUDA Block Size](images/framerate_vs_block_size.png)



### Analysis

Changing the block size changes both the number of threads grouped into each CUDA block and, for a fixed number of boids, the number of blocks required to cover the simulation.

We see that the **naive implementation** performed best around 64-128 threads per block in these measurements. Its mean FPS increased from 187.16 at a block size of 32 to 295.10 at 64, remained similar at 128, and then gradually declined to 274.44 at 512.


We also see that the **scattered and coherent grid implementations were considerably more non-monotonic**. For instance, the scattered implementation measured 1428.70 FPS at a block size of 32, fell to 895.38 at 64 and 917.74 at 128, then increased to 1614.36 at 256. The coherent implementation similarly reached its lowest measured value at 128 threads and its highest at 512 threads.


I therefore was not able to identify one block size that was best for all three implementations. The naive implementation was relatively stable from 64 through 256 threads, while the grid implementations showed much larger non-monotonic changes. A grid-based simulation step contains several different operations, including grid-index computation, sorting, cell-range construction, neighbor searching, and, for the coherent version, data reordering. Changing the global block size therefore affects several CUDA kernels with different workloads rather than one uniform computation. Since I measured the complete application frame, these results show the overall performance effect of changing block size but do not isolate which kernel is responsible for each peak or decline.


## 3. Semi-Coherent Uniform Grid Performance

The semi-coherent grid did produce performance improvements over the scattered grid at some boid counts, but **the improvement was not consistent across the complete benchmark range**.


With visualization disabled, we measured these:

| Boids | Scattered FPS | Coherent FPS | Faster Implementation |
|---:|---:|---:|:---|
| 1,000 | 2360.45 | 1308.48 | Scattered |
| 2,500 | 1922.68 | 1225.16 | Scattered |
| 5,000 | 1759.70 | 1116.78 | Scattered |
| 10,000 | 1733.18 | 1031.66 | Scattered |
| 20,000 | 917.74 | 1153.78 | Coherent |
| 40,000 | 755.78 | 1672.40 | Coherent |



At 20,000 boids, the coherent implementation was approximately **25.72% faster** than the scattered implementation in the visualization-off measurements. At 40,000 boids, the measured difference was substantially larger. However, from 1,000 through 10,000 boids, the scattered implementation was faster.


I expected the semi-coherent representation to have the potential to improve neighbor-search performance as the reordered position and velocity arrays place boids from the same grid cells contiguously in memory. The scattered implementation instead uses the sorted particle-index array as an additional indirection into the original position and velocity arrays.

However, coherence is not free: the semi-coherent implementation must reorder the boid data after sorting during every simulation step. Therefore, improved memory locality during the neighbor search must compensate for the additional reordering work. My measurements suggest that this tradeoff did not consistently favor the coherent implementation at smaller and medium population sizes.


The 40,000-boid coherent result is unusually high relative to the rest of its curve, so I retained it rather than smoothing it away. The overall crossover is nevertheless consistent with the expected tradeoff: reordering introduces additional work every frame, while its benefit comes from improving locality during the subsequent neighbor search.



## 4. Cell Width: 8 Vs. 27 Neighboring Cells

I compared the scattered uniform-grid implementation using two grid-cell widths.


Our baseline used a cell width of twice the maximum interaction radius (`2R`). With this cell width, the neighbor search needs to inspect at most 8 relevant cells in 3D.


I then changed the cell width to the interaction radius (`R`) and searched the full 3 x 3 x 3 neighborhood (or 27 cells).


The comparison used **20,000 boids**, `VISUALIZE = 0`, a block size of 128, Release x64, and Vertical Sync disabled.

| Grid Configuration | Neighboring Cells Checked | Mean FPS |
|:---|---:|---:|
| Cell width = `2R` | Up to 8 | 917.74 |
| Cell width = `R` | 27 | 1105.44 |



The measured speedup of the 27-cell configuration was approximately this:

`(1105.44 / 917.74 - 1) * 100 = 20.45%`

Thus, the 27-cell configuration was approximately **20.45% faster** than the 8-cell configuration in this experiment.


This result demonstrates why the number of cells checked alone is not sufficient to predict performance. Reducing the cell width increases the number of cell ranges that must be visited, but it also makes each cell spatially smaller. Smaller cells can contain fewer candidate boids, which can reduce the number of unnecessary candidate-distance and rule checks during the neighbor search.


For this benchmark, the measured result is consistent with the reduction in candidate-neighbor work outweighing the additional cost of traversing 27 cell ranges. However, I did not directly instrument the number of candidate comparisons, so this is a hypothesis consistent with the algorithm and observed performance rather than a separately measured causal result.


## Build Instructions

I built and tested this project on Windows 11 using Visual Studio 2026 and CUDA 13.3.

From the project repository, create and enter a build directory:

```bash
mkdir build
cd build
```

Then configure the project with CMake:

```bash
cmake .. -G "Visual Studio 18 2026" -A x64
```

Open the generated Visual Studio solution and build the `cis5650_boids` project using the `Release | x64` configuration.


# Extra Credit

The required implementation and all performance measurements above were completed **before** attempting extra-credit optimizations.

Extra-credit implementation details and any additional performance comparisons will be documented here separately so that they can be distinguished from the required baseline results.

<!-- Extra-credit results to be added after implementation and testing. -->
