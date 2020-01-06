# Benchmarks

The performance of PencilFFTs.jl is comparable to that of other open-source
parallel FFT libraries implemented in lower-level languages.
Below, we show comparisons with the Fortran implementation of
[P3DFFT](https://www.p3dfft.net/), possibly the most popular of these
libraries.
The benchmarks were performed on the [Jean--Zay
cluster](http://www.idris.fr/jean-zay/jean-zay-presentation.html) of the IDRIS
French computing centre
([description in
English](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html)).

The figure below shows [strong
scaling](https://en.wikipedia.org/wiki/Scalability#Weak_versus_strong_scaling)
benchmarks of 3D real-to-complex FFTs using 2D ("pencil") decomposition.
The benchmarks were run for input arrays of dimensions
$N_x × N_y × N_z = 512^3$ and $1024^3$.
Each timing is averaged over 100 repetitions.

```@raw html
<div class="figure">
  <!--
  Note: this is evaluated from the directory where the Benchmarks page is
  built. This directory varies depending on whether `prettyurls` is enabled in
  `makedocs`. Here we assume `prettyurls=true`.
  -->
  <img
    width="75%"
    src="../img/benchmark_idris.svg"
    alt="Strong scaling of PencilFFTs">
</div>
```

The performance and scalability of PencilFFTs are similar to those displayed
by P3DFFT.
In general, P3DFFT has a small advantage over PencilFFTs, which is possibly explained because their local array reordering routines are better optimised.
In the future we expect to close that performance gap.

On the other hand, an important difference is that PencilFFTs uses non-blocking
point-to-point MPI communications by default (using `MPI_Isend` and
`MPI_Irecv`), while P3DFFT uses global `MPI_Alltoallv` calls.
This enables us to perform data reordering operations on the partially received
data while we wait for the incoming data, and thus can lead to better
performance especially when running on a large number of processes.

Note that PencilFFTs can optionally use `MPI_Alltoallv` instead of
point-to-point communications (see the docs for [`PencilFFTPlan`](@ref) and
[`transpose!`](@ref)).
We have verified that the implementation with `MPI_Isend` and `MPI_Irecv` generally
outperforms the one based on `MPI_Alltoallv`.
Observed performance gains can be of the order of 10%.

### Benchmark details

The benchmarks were performed using Julia 1.3.1 and Intel MPI 2019.0.4.
We used PencilFFTs v0.2.0 with FFTW.jl v1.2.0 and MPI.jl v0.11.0.
We use the Fortran implementation of P3DFFT, version 2.7.6.
P3DFFT v2.7.6 (Fortran version) was built with Intel 2019 compilers and linked
to FFTW 3.3.8.
The cluster where the benchmarks were run has Intel Cascade Lake 6248
processors with 2×20 cores per node.

The number of MPI processes along each decomposed dimension, $P_1$ and $P_2$,
was automatically determined by a call to `MPI_Dims_create`,
which tends to create a balanced decomposition with $P_1 ≈ P_2$.
For instance, a total of 1024 processes is divided into $P_1 = P_2 = 32$.
Different results may be obtained with other combinations, but this was not
benchmarked.

The source files used to generate this benchmark, as well as the raw benchmark
results, are all available [in the
PencilFFTs repo](https://github.com/jipolanco/PencilFFTs.jl/tree/master/benchmarks/clusters/idris.jean_zay).
