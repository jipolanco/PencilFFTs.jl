# Tutorial

```@meta
CurrentModule = PencilFFTs
```

The following tutorial shows how to perform a 3D FFT of real periodic data
defined on a grid of $N_x × N_y × N_z$ points.

```@raw html
<div class="figure">
  <!--
  Note: this is evaluated from the directory where the Tutorial page is
  built. This directory varies depending on whether `prettyurls` is enabled in
  `makedocs`. Here we assume `prettyurls=true`.
  -->
  <img
    width="85%"
    src="../img/pencils.svg"
    alt="Pencil decomposition of 3D domains">
</div>
```

By default, the domain is distributed on a 2D MPI topology of dimensions
``N_1 × N_2``.
As an example, the above figure shows such a topology with ``N_1 = 4`` and
``N_2 = 3``, for a total of 12 MPI processes.

## [Creating plans](@id tutorial:creating_plans)

The first thing to do is to create a domain decomposition configuration for the
given dataset dimensions ``N_x × N_y × N_z``.
In the framework of PencilArrays, such a configuration is described by
a `Pencil` object.
As described in the [PencilArrays
docs](https://jipolanco.github.io/PencilArrays.jl/dev/Pencils/), we can let the
`Pencil` constructor automatically determine such a configuration.
For this, only an MPI communicator and the dataset dimensions are needed:

```julia
using MPI
using PencilFFTs

MPI.Init()
comm = MPI.COMM_WORLD

# Input data dimensions (Nx × Ny × Nz)
dims = (16, 32, 64)
pen = Pencil(dims, comm)
```

By default this creates a 2D decomposition (for the case of a 3D dataset), but
one can change this as detailed in the PencilArrays documentation linked above.

We can now create a [`PencilFFTPlan`](@ref), which requires
information on decomposition configuration (the `Pencil` object) and on the
transforms that will be applied:

```julia
# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# Note that, for more control, one can instead separately specify the transforms along each dimension:
# transform = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())

# Create plan
plan = PencilFFTPlan(pen, transform)
```

See the [`PencilFFTPlan`](@ref) constructor for details on the accepted
options, and the [`Transforms`](@ref) module for the possible transforms.
It is also possible to enable fine-grained performance measurements via the
[TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) package, as
described in [Measuring performance](@ref PencilFFTs.measuring_performance).

## Allocating data

Next, we want to apply the plan on some data.
Transforms may only be applied on
[`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/)s,
which are array
wrappers that include MPI decomposition information (in some sense, analogous
to [`DistributedArray`](https://github.com/JuliaParallel/Distributedarrays.jl)s
in Julia's distributed computing approach).
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

The data types returned by [`allocate_input`](@ref) and
[`allocate_output`](@ref) are slightly different when working with in-place
transforms.
See the [in-place example](@ref In-place-transforms) for details.

## Applying plans

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

## Accessing and modifying data

For any given MPI process, a `PencilArray` holds the data associated to its
local partition in the global geometry.
`PencilArray`s are accessed using local indices that start at 1, regardless of
the location of the local process in the MPI topology.
Note that `PencilArray`s, being based on regular `Array`s, support both linear
and Cartesian indexing (see [the Julia
docs](https://docs.julialang.org/en/v1/manual/arrays/#man-supported-index-types)
for details).

For convenience, the [`global_view`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#Global-views) function can be used to generate an
[`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) wrapper that
takes global indices.

### [Output data layout](@id tutorial:output_data_layout)

In memory, the dimensions of the transform output are by default reversed with
respect to the input.
That is, if the order of indices in the input data is `(x, y, z)`, then the
output has order `(z, y, x)` in memory.
This detail is hidden from the user, and **output arrays are always accessed in
the same order as the input data**, regardless of the underlying output
dimension permutation.
This applies to `PencilArray`s and to `OffsetArray`s returned by
[`global_view`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.global_view-Tuple{PencilArray}).

The reasoning behind dimension permutations, is that they allow to always
perform FFTs along the fastest array dimension and to avoid a local data
transposition, resulting in performance gains.
A similar approach is followed by other parallel FFT libraries.
FFTW itself, in its distributed-memory routines, [includes
a flag](http://fftw.org/doc/Transposed-distributions.html#Transposed-distributions)
that enables a similar behaviour.
In PencilFFTs, index permutation is the default, but it can be disabled via the
`permute_dims` flag of [`PencilFFTPlan`](@ref).

A great deal of work has been spent in making generic index permutations as
efficient as possible, both in intermediate and in the output state of the
multidimensional transforms.
This has been achieved, in part, by making sure that permutations such as `(3,
2, 1)` are compile-time constants.

## Further reading

For details on working with `PencilArray`s see the
[PencilArrays docs](https://jipolanco.github.io/PencilArrays.jl/dev/).

The examples on the sidebar further illustrate the use of transforms and
provide an introduction to working with MPI-distributed data in the form of
`PencilArray`s.
In particular, the [gradient example](@ref Gradient-of-a-scalar-field)
illustrates different ways of computing things using Fourier-transformed
distributed arrays.
Then, the [incompressible Navier--Stokes example](@ref Navier–Stokes-equations)
is a more advanced and complete example of a possible application of the
PencilFFTs package.
