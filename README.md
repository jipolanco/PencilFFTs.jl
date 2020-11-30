# PencilFFTs

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://jipolanco.github.io/PencilFFTs.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://jipolanco.github.io/PencilFFTs.jl/dev/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3618781.svg)](https://doi.org/10.5281/zenodo.3618781)

[![Build Status](https://github.com/jipolanco/PencilFFTs.jl/workflows/CI/badge.svg)](https://github.com/jipolanco/PencilFFTs.jl/actions)
[![Coverage](https://codecov.io/gh/jipolanco/PencilFFTs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jipolanco/PencilFFTs.jl)

Fast Fourier transforms of MPI-distributed Julia arrays.

This package provides multidimensional FFTs and related transforms on
MPI-distributed Julia arrays via the
[PencilArrays](https://github.com/jipolanco/PencilArrays.jl) package.

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below,
where each coloured block is managed by a different MPI process.
Typically, one wants to compute FFTs on a scalar or vector field along the
three spatial dimensions.
In the case of a pencil decomposition, 3D FFTs are performed one dimension at
a time (along the non-decomposed direction, using a serial FFT implementation).
Global data transpositions are then needed to switch from one pencil
configuration to the other and perform FFTs along the other dimensions.

<p align="center">
  <br/>
  <img width="85%" alt="Pencil decomposition of 3D domains" src="docs/src/img/pencils.svg">
</p>

## Features

- distributed `N`-dimensional FFTs of MPI-distributed Julia arrays, using
  the [PencilArrays](https://github.com/jipolanco/PencilArrays.jl) package;

- FFTs and related transforms (e.g.
  [DCTs](https://en.wikipedia.org/wiki/Discrete_cosine_transform) / Chebyshev
  transforms) may be arbitrarily combined along different dimensions;

- in-place and out-of-place transforms;

- high scalability up to (at least) tens of thousands of MPI processes.

## Installation

PencilFFTs can be installed using the Julia package manager:

    julia> ] add PencilFFTs

## Quick start

The following example shows how to apply a 3D FFT of real data over 12 MPI
processes distributed on a `3 × 4` grid (same distribution as in the figure
above).

```julia
using MPI
using PencilFFTs
using Random

MPI.Init()

dims = (16, 32, 64)  # input data dimensions
transform = Transforms.RFFT()  # apply a 3D real-to-complex FFT

# Distribute 12 processes on a 3 × 4 grid.
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
proc_dims = (3, 4)

# Create plan
plan = PencilFFTPlan(dims, transform, proc_dims, comm)

# Allocate and initialise input data, and apply transform.
u = allocate_input(plan)
rand!(u)
uF = plan * u

# Apply backwards transform. Note that the result is normalised.
v = plan \ uF
@assert u ≈ v
```

For more details see the
[tutorial](https://jipolanco.github.io/PencilFFTs.jl/dev/tutorial/).

## Performance

The performance of PencilFFTs is comparable to that of widely adopted MPI-based
FFT libraries implemented in lower-level languages.
As seen below, with its default settings, PencilFFTs generally outperforms the Fortran [P3DFFT](https://www.p3dfft.net/) libraries.

<p align="center">
  <br/>
  <img width="70%" alt="Strong scaling of PencilFFTs" src="docs/src/img/benchmark_idris.svg">
</p>

See [the benchmarks
section](https://jipolanco.github.io/PencilFFTs.jl/dev/benchmarks/) of the docs
for details.
