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
This is illustrated by the figure below
([source](https://hal.archives-ouvertes.fr/tel-02084215v1)),
where each coloured block is managed by a different MPI process.
Typically, one wants to compute FFTs on a scalar or vector field along the
three spatial dimensions.
In the case of a pencil decomposition, 3D FFTs are performed one dimension at
a time (along the non-decomposed direction, using a serial FFT implementation).
Global data transpositions are then needed to switch from one pencil
configuration to the other and perform FFTs along the other dimensions.

```@raw html
<div class="figure">
  <img src="img/pencils.svg" alt="Pencil decomposition of 3D domains">
</div>
```

The package is implemented in an efficient generic way that allows to decompose
any `N`-dimensional geometry along `M < N` dimensions (for the pencil
decomposition described above, `N = 3` and `M = 2`). Moreover, the transforms
applied along each dimension can be arbitrarily chosen among those supported by
`FFTW`, including complex-to-complex, real-to-complex and
real-to-real transforms.

## Getting started

Say you want to perform a 3D FFT of real periodic data defined on
``N_x × N_y × N_z`` grid points.
The data is to be distributed over 12 MPI processes on a ``3 × 4`` grid, as in
the figure above.

### Creating plans

The first thing to do is to create a [`PencilFFTPlan`](@ref), which requires
information on the global dimensions ``N_x × N_y × N_z`` of the data, on the
transforms that will be applied, and on the way the data is distributed among
MPI processes (the MPI Cartesian topology):

```julia
using MPI
using PencilFFTs

MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
dims = (16, 32, 64)

# Apply a real-to-complex (r2c) FFT
transform = Transforms.RFFT()

# MPI topology information
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
proc_dims = (3, 4)     # 3×4 Cartesian topology

# Create plan
plan = PencilFFTPlan(dims, transform, proc_dims, comm)
```

See the [`PencilFFTPlan`](@ref) constructor for details on the accepted
options, and the [`Transforms`](@ref) module for the possible transforms.
It is also possible to enable fine-grained performance measurements via the
[TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) package, as
described in [Measuring performance](@ref PencilFFTs.measuring_performance).

### Allocating data

Next, we want to apply the plan on some data.
Transforms may only be applied on [`PencilArray`](@ref)s, which are array
wrappers that include MPI decomposition information.
The helper function [`allocate_input`](@ref) may be used to allocate
a `PencilArray` that is compatible with our plan:
```julia
# In our example, this returns a 3D PencilArray of real data (Float64).
u = allocate_input(plan)

# Fill the array with some (random) data
using Random
randn!(u)
```
`PencilArray`s are a subtype of `AbstractArray`, and thus they support all
common array operations.

Similarly, to preallocate output data, one can use [`allocate_output`](@ref):
```julia
# In our example, this returns a 3D PencilArray of complex data (Complex{Float64}).
v = allocate_output(plan)
```
This is only required if one wants to apply the plans using a preallocated
output (with `mul!`, see below).

### Applying plans

The interface to apply plans is consistent with that of
[`AbstractFFTs`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft).
Namely, `*` and `mul!` are respectively used for forward transforms without and
with preallocated output data.
Similarly, `\ ` and `ldiv!` are used for backward transforms.

```julia
using LinearAlgebra  # for mul!, ldiv!

# Apply plan on `u` with `v` as an output
mul!(v, plan, u)

# Apply backward plan on `v` with `w` as an output
w = similar(u)
ldiv!(w, plan, v)  # now w ≈ u
```

Note that, consistently with `AbstractFFTs`, backward transforms are
normalised, so that the original data is recovered (possibly up to a very small
error) when applying a forward followed by a backward transform.

Also note that, at this moment, in-place transforms are not supported.

## Working with MPI-distributed data

The implementation of this package is modular.
Distributed FFTs are built on top of the [`Pencils`](@ref Pencils_module)
module that handles
data decomposition among MPI processes, including the definition of relevant
data structures and global data transpositions.
The data decomposition functionality may be used independently of the
FFTs.
See the [`Pencils`](@ref Pencils_module) module documentation for more details.

## Similar projects

- [PFFT](https://www-user.tu-chemnitz.de/~potts/workgroup/pippig/software.php.en#pfft)
  is a very general parallel FFT library written in C.

- [P3DFFT](https://www.p3dfft.net) implements parallel 3D FFTs using pencil
  decomposition in Fortran and C++.

- [2DECOMP&FFT](http://www.2decomp.org) is another parallel 3D FFT library
  using pencil decomposition written in Fortran.
