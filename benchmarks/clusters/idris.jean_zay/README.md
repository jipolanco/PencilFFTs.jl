# Benchmarks on the Jean-Zay cluster

[Strong
scaling](https://en.wikipedia.org/wiki/Scalability#Weak_versus_strong_scaling)
benchmarks of 3D real-to-complex FFTs using a 2D ("pencil") decomposition.

The number of MPI processes along each dimension (`P1` and `P2`) is
automatically determined by `MPI_Dims_create`.
In our tests, MPI tends to create a balanced decomposition with `P1 ≈ P2`.
For instance, a total of 1024 processes is divided into `P1 = P2 = 32`.

![Strong scaling of PencilFFTs](timing_comparison.svg)

## Machine

Tests run on the [Jean-Zay cluster](http://www.idris.fr/jean-zay/jean-zay-presentation.html)
([English version](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html)) of
IDRIS (France).

Some relevant specifications:

- HPE SGI 8600 computer
- 1528 XA730i compute nodes, with 2 Intel Cascade Lake 6248 processors (20
  cores at 2.5 GHz), or 61120 cores available

## Software

The benchmarks were run using Julia 1.3, Intel MPI 2019.0.4 and FFTW 3.3.9
(version bundled by FFTW.jl).

## Version

Date: 4 December 2019 (PencilFFTs commit
[`3f238172a`](https://github.com/jipolanco/PencilFFTs.jl/commit/3f238172a62036104f535d7bb22933096458f9a8)).
