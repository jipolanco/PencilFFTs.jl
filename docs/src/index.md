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

## Example usage

Say you want to perform a 3D FFT of real periodic data defined on
$N_x × N_y × N_z$ grid points.
The data is to be distributed over 12 MPI processes on a $3 × 4$ grid, as in
the figure above.

### Creating plans

The first thing to do is to create a [`PencilFFTPlan`](@ref), which requires
information on the global dimensions $N_x × N_y × N_z$ of the data, on the
transforms that will be applied, and on the way the data is distributed among
MPI processes (i.e. number of processes along each dimension):

```julia
using MPI
using PencilFFTs

MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
dims = (16, 32, 64)

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# For more control, one can instead separately specify the transforms along each dimension:
# transform = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())

# MPI topology information
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
proc_dims = (3, 4)     # 3 processes along `y`, 4 along `z`

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
wrappers that include MPI decomposition information (in some sense, analogous
to [`DistributedArray`](https://github.com/JuliaParallel/Distributedarrays.jl)s
in Julia's native distributed programming approach).
The helper function [`allocate_input`](@ref) can be used to allocate
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
output (with `mul!`, see right below).

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

Note that, consistently with `AbstractFFTs`,
normalisation is performed at the end of a backward transform, so that the
original data is recovered when applying a forward followed by a backward
transform.

Also note that, at this moment, in-place transforms are not supported.

### Accessing and modifying data

For any given MPI process, a `PencilArray` holds the data associated to its
local partition in the global geometry.
`PencilArray`s are accessed using local indices that start at 1, regardless of
the location of the local process in the MPI topology.
Note that `PencilArray`s, being based on regular `Array`s, support both linear
and Cartesian indexing (see [the Julia
docs](https://docs.julialang.org/en/latest/manual/arrays/#Number-of-indices-1)
for details).

For convenience, the [`global_view`](@ref) function can be used to generate an
[`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) wrapper that
takes global indices.

Finally note that, by default, the output of a multidimensional transform
should be accessed with permuted indices.
That is, if the order of indices in the input data is `(x, y, z)`, then the
output has order `(z, y, x)`.
This allows to always perform FFTs along the fastest array dimension and
to avoid a local data transposition, resulting in performance gains.
A similar approach is followed by other parallel FFT libraries
(FFTW itself, in its distributed-memory routines, [includes
a flag](http://fftw.org/doc/Transposed-distributions.html#Transposed-distributions)
that enables a similar behaviour).
In PencilFFTs, index permutation is the default, but it can be disabled via the
`permute_dims` flag of [`PencilFFTPlan`](@ref).
As a side note, a great deal of work has been spent in making generic index
permutations as efficient as possible, both in intermediate and in the output state of the multidimensional transforms.
This has been achieved, in part, by making sure that permutations such as `(3,
2, 1)` are compile-time constants (i.e. [value
types](https://docs.julialang.org/en/latest/manual/types/#%22Value-types%22-1)).
In the future, permutations will probably be performed invisibly, without the
user having to care about them.

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
Pages = ["PencilFFTs.md", "Transforms.md", "Pencils.md"]
Depth = 2
```

[^1]:
    Figure adapted from [this thesis](https://hal.archives-ouvertes.fr/tel-02084215v1).

[^2]:
    For the pencil decomposition represented in the figure, $N = 3$ and $M = 2$.
