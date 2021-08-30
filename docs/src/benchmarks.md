# Benchmarks

The performance of PencilFFTs.jl is comparable to that of other open-source
parallel FFT libraries implemented in lower-level languages.
Below, we show comparisons with the Fortran implementation of
[P3DFFT](https://www.p3dfft.net/), possibly the most popular of these
libraries.
The benchmarks were performed on the [Jean--Zay
cluster](http://www.idris.fr/jean-zay/jean-zay-presentation.html) of the IDRIS
French computing centre (CNRS).

The figure below shows [strong
scaling](https://en.wikipedia.org/wiki/Scalability#Weak_versus_strong_scaling)
benchmarks of 3D real-to-complex FFTs using 2D ("pencil") decomposition.
The benchmarks were run for input arrays of dimensions
$N_x × N_y × N_z = 512^3$, $1024^3$ and $2048^3$.
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

As seen above, PencilFFTs generally outperforms P3DFFT in its default setting.
This is largely explained by the choice of using non-blocking point-to-point
MPI communications (via
[`MPI_Isend`](https://www.open-mpi.org/doc/current/man3/MPI_Isend.3.php) and
[`MPI_Irecv`](https://www.open-mpi.org/doc/current/man3/MPI_Irecv.3.php)),
while P3DFFT uses collective
[`MPI_Alltoallv`](https://www.open-mpi.org/doc/current/man3/MPI_Alltoallv.3.php)
calls.
This enables PencilFFTs to perform data reordering operations on the partially received data while waiting for the incoming data, leading to better performance.
Moreover, in contrast with P3DFFT, the high performance and scalability of
PencilFFTs results from a highly generic code, handling decompositions in
arbitrary dimensions and a relatively large (and extensible) variety of
transformations.

Note that PencilFFTs can optionally use collective communications (using
`MPI_Alltoallv`) instead of point-to-point communications.
For details, see the docs for [`PencilFFTPlan`](@ref) and
for [`PencilArray` transpositions](https://jipolanco.github.io/PencilArrays.jl/dev/Transpositions/#PencilArrays.Transpositions.Transposition).
As seen above, collective communications generally perform worse than point-to-point ones, and runtimes are nearly indistinguishable from those of P3DFFT.

### Benchmark details

The benchmarks were performed using Julia 1.7-beta3 and Intel MPI 2019.
We used PencilFFTs v0.12.5 with FFTW.jl v1.4.3 and MPI.jl v0.19.0.
We used the Fortran implementation of P3DFFT, version 2.7.6,
which was built with Intel 2019 compilers and linked to FFTW 3.3.8.
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
