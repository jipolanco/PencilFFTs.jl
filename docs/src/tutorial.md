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

The example assumes that 12 MPI processes are available.
The data is to be distributed on a 2D MPI topology of dimensions $3 × 4$,
as represented in the above figure.

## [Creating plans](@id tutorial:creating_plans)

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

## Allocating data

Next, we want to apply the plan on some data.
Transforms may only be applied on [`PencilArray`](@ref)s, which are array
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
docs](https://docs.julialang.org/en/latest/manual/arrays/#Number-of-indices-1)
for details).

For convenience, the [`global_view`](@ref) function can be used to generate an
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
[`global_view`](@ref).

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

## Parallel I/O

It is possible to read and write `PencilArray`s to disk using [Parallel
HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5) via the
[HDF5.jl](https://github.com/JuliaIO/HDF5.jl) package.
Assuming [everything is set-up correctly](@ref setting_up_parallel_hdf5), the
following code writes the arrays `u` and `v` created in the previous sections:

```julia
using HDF5
using MPI
using PencilFFTs

#= code from previous sections... =#

info = MPI.Info()

ph5open("fields.h5", "w", comm, info) do ff
    ff["u"] = u
    ff["v"] = v
end
```

Note that, for this to work, the code in [Creating plans](@ref
tutorial:creating_plans) must be modified so that HDF5 is loaded before
PencilFFTs, since HDF5 is lazy-loaded using
[Requires](https://github.com/JuliaPackaging/Requires.jl).

The following code reads back the datasets into `PencilArray`s:

```julia
u_read = similar(u)
v_read = similar(v)

ph5open("fields.h5", "r", comm, info) do ff
    read!(ff, u_read, "u")
    read!(ff, v_read, "v")
end

@assert u == u_read
@assert v == v_read
```

### Data layout in files

In our example, the `v` array, being the output of a distributed transform, has
its dimensions permuted in memory.
This was discussed in [Output data layout](@ref tutorial:output_data_layout).
For performance reasons, data is written to the HDF5 file in memory order, and
thus the dimensions of the `v` array in the file are permuted with respect to
those of the input array.
This detail is important if one wishes to read the data from a different
application.

### More documentation

The Parallel I/O feature is documented in its [dedicated section](@ref
PencilIO_module).
In particular, a short step-by-step guide is provided on how to set-up the
MPI.jl and HDF5.jl packages in order to make things work.
This is not trivial, as it requires the HDF5 libraries to be built with
parallel support, and linked to same MPI libraries used by MPI.jl.

## Further reading

The examples on the sidebar further illustrate the use of transforms and
provide an introduction to working with MPI-distributed data in the form of
`PencilArray`s.

In addition to the examples,
some useful scripts are available in the `test/` directory of the
`PencilFFTs` repo.
In particular, the
[`test/taylor_green.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/test/taylor_green.jl)
example is a (very simple) fluid dynamics application around the
[Taylor--Green](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex)
vortex flow.
