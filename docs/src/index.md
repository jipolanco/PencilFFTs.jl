# PencilFFTs.jl

```@meta
CurrentModule = PencilFFTs
```

Fast Fourier transforms of MPI-distributed Julia arrays.

## Introduction

This package provides functionality to distribute multidimensional arrays among
MPI processes, and to perform multidimensional FFTs (and related transforms) on
them.

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below,[^1]
where each coloured block is managed by a different MPI process.
Typically, one wants to compute FFTs on a scalar or vector field along the
three spatial dimensions.
In the case of a pencil decomposition, 3D FFTs are performed one dimension at
a time, along the non-decomposed direction.
Transforms must then be interleaved with global data transpositions to switch
between pencil configurations.
In high-performance computing environments, such data transpositions are
generally the most expensive part of a parallel FFT computation, due to the
large cost of communications between computing nodes.

```@raw html
<div class="figure">
  <img
    width="85%"
    src="img/pencils.svg"
    alt="Pencil decomposition of 3D domains">
</div>
```

More generally, PencilFFTs allows to decompose and perform FFTs on geometries
of arbitrary dimension $N$.
The decompositions can be performed along an arbitrary number $M < N$ of
dimensions.[^2]
Moreover, the transforms applied along each dimension can be arbitrarily chosen
among those supported by [FFTW.jl](https://github.com/JuliaMath/FFTW.jl),
including complex-to-complex, real-to-complex and real-to-real transforms.
The generic and efficient implementation of this package is greatly enabled by
the use of zero-cost abstractions in Julia.
As shown in the [Benchmarks](@ref) section, the performance of PencilFFTs has
been validated against the C++ implementation of the
[P3DFFT](https://www.p3dfft.net) library.

## Similar projects

- [FFTW3](http://fftw.org/doc/Distributed_002dmemory-FFTW-with-MPI.html#Distributed_002dmemory-FFTW-with-MPI)
  implements distributed-memory transforms using MPI, but these are limited to
  1D decompositions.
  Also, this functionality is not currently included in the FFTW.jl wrappers.

- [PFFT](https://www-user.tu-chemnitz.de/~potts/workgroup/pippig/software.php.en#pfft)
  is a very general parallel FFT library written in C.

- [P3DFFT](https://www.p3dfft.net) implements parallel 3D FFTs using pencil
  decomposition in Fortran and C++.

- [2DECOMP&FFT](http://www.2decomp.org) is another parallel 3D FFT library
  using pencil decomposition written in Fortran.

## Contents

```@contents
Pages = [
    "tutorial.md",
    "PencilFFTs.md",
    "Transforms.md",
    "Pencils.md",
    "benchmarks.md",
]
Depth = 2
```

[^1]:
    Figure adapted from [this thesis](https://hal.archives-ouvertes.fr/tel-02084215v1).

[^2]:
    For the pencil decomposition represented in the figure, $N = 3$ and $M = 2$.
