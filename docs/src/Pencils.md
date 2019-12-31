# [MPI-distributed data](@id Pencils_module)

```@meta
CurrentModule = PencilFFTs.Pencils
```

The distribution of global data among MPI processes is managed by the [`Pencils`](@ref) module.
This module may be used independently of the FFT functionality.

The [`Pencils`](@ref) module defines types that describe [an MPI Cartesian
topology](@ref sec:mpi_topology) and [the decomposition of data over MPI
processes](@ref sec:pencil_configs).
The module also defines [array wrappers](@ref Array-wrappers), most notably the
[`PencilArray`](@ref) type, which allow to conveniently and efficiently work
with MPI-decomposed data.

```@raw html
<!-- TODO Summarise sections. Skip bla bla if you're not interested. -->
```

## [MPI topology](@id sec:mpi_topology)

The [`MPITopology`](@ref) type defines the MPI Cartesian topology of the
decomposition.
In other words, it contains information about the number of decomposed
dimensions, and the number of processes in each of these dimensions.

The main `MPITopology` constructor takes a MPI communicator and a tuple
specifying the number of processes in each dimension.
For instance, to distribute 12 MPI processes on a $3 × 4$ grid:
```julia
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
pdims = (3, 4)
topology = MPITopology(comm, pdims)
```

At the lower level, [`MPITopology`](@ref) uses
[`MPI_Cart_create`](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html)
to define a Cartesian MPI communicator.
For more control, one can also create a Cartesian communicator using
`MPI.Cart_create`, and pass that to `MPITopology`:
```julia
comm = MPI.COMM_WORLD
dims = [3, 4]  # note: array, not tuple!
periods = zeros(Int, N)
reorder = false
comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
topology = MPITopology(comm_cart)
```

## [Pencil configurations](@id sec:pencil_configs)

A *pencil* configuration refers to a given distribution of multidimensional
data among MPI processes.
This information is encoded in the [`Pencil`](@ref) type.

More precisely, a pencil configuration includes:
- [MPI topology](@ref sec:mpi_topology) information,
- global and local dimensions of the numerical grid,
- subset of decomposed dimensions,
- type of decomposed data (e.g. `Float64`),
- definition of optional permutation of dimensions.

### Construction

The creation of a new [`Pencil`](@ref) requires a [`MPITopology`](@ref), as
well as the global data dimensions and a list of decomposed dimensions.
Optionally, one can also specify the data type (the default is `Float64`) and
a permutation of dimensions.

For instance, to decompose along the first and third dimensions of a complex
3D dataset,
```julia
topology = MPITopology(#= ... =#)
dims_global = (16, 32, 64)
decomp_dims = (1, 3)  # this requires ndims(topology) == 2
pencil = Pencil(topology, dims_global, decomp_dims, Complex{Float64})
```

One may also want to create multiple pencil configurations that differ, for
instance, on the selection of decomposed dimensions.
For this case, a second constructor is available that takes an already existing
`Pencil` instance.
Calling this constructor should be preferred when possible since it allows
sharing memory buffers (used for instance for [global transpositions](@ref Global-MPI-operations)) and thus reducing memory usage.
The following creates a `Pencil` equivalent to the one above, but with
different decomposed dimensions:
```julia
pencil_x = Pencil(pencil, decomp_dims=(2, 3))
```
See the [`Pencil`](@ref) documentation for more details.

### Dimension permutations

As mentioned above, a `Pencil` may optionally be given information on dimension
permutations.
In this case, the layout of the data arrays in memory is different from the
logical order of dimensions.

To make this clearer, consider the example above where the global data
dimensions are $N_x × N_y × N_z = 16 × 32 × 64$.
In this case, the logical order is $(x, y, z)$.
Now let's say that we want the memory order of the data to be $(y, z, x)$,[^1]
which corresponds to the permutation `(2, 3, 1)`.

Permutations are passed to the `Pencil` constructor via the `permute` keyword
argument.
For performance reasons, in the `Pencils` module, dimension permutations are
compile-time constants, and thus permutations should be specified as [value
types](https://docs.julialang.org/en/latest/manual/types/#%22Value-types%22-1)
wrapping a tuple.
For instance,
```julia
permutation = Val((2, 3, 1))
pencil = Pencil(#= ... =#, permute=permutation)
```
One can also pass `nothing` as a permutation, which disables permutations (this
is the default).

## Array wrappers

### Types

```@docs
PencilArray
PencilArrayCollection
MaybePencilArrayCollection
```

### Functions

```@docs
extra_dims(::PencilArray)
get_comm(::MaybePencilArrayCollection)
get_permutation(::MaybePencilArrayCollection)
global_view(::PencilArray)
ndims_extra(::MaybePencilArrayCollection)
ndims_space(::PencilArray)
parent(::PencilArray)
pencil(::PencilArray)
range_local(::MaybePencilArrayCollection)
size(::PencilArray)
size_global(::MaybePencilArrayCollection)
spatial_indices(::PencilArray)
```

## Global MPI operations

One of the most time-consuming parts of a large-scale computation involving
multidimensional FFTs, is the global data transpositions between different MPI
decomposition configurations.
In `Pencils`, this is performed by the [`transpose!`](@ref) function, which
takes two `PencilArray`s, typically associated to two different configurations.
The implementation performs comparably to similar implementations in
lower-level languages (see [Benchmarks](@ref)).

Also provided is a [`gather`](@ref) function that creates a single global array
from decomposed data.
This can be useful for tests (in fact, it is used in the `Pencils` tests to
verify the correctness of the transpositions), but shouldn't be used with large
datasets.
It is generally useful for small problems where the global size of the data can
easily fit the locally available memory.

```@docs
transpose!
gather
```

## [Measuring performance](@id Pencils.measuring_performance)

It is possible to measure the time spent in different sections of the MPI data transposition routines using the [TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) package. This has a (very small) performance overhead, so it is disabled by default. To enable time measurements, call `TimerOutputs.enable_debug_timings(PencilFFTs.Pencils)` after loading `PencilFFTs`. For more details see the [TimerOutputs docs](https://github.com/KristofferC/TimerOutputs.jl#overhead).

Minimal example:

```julia
using MPI
using PencilFFTs.Pencils
using TimerOutput

# Enable timing of `Pencils` functions
TimerOutputs.enable_debug_timings(PencilFFTs.Pencils)

MPI.Init()

pencil = Pencil(#= args... =#)

# [do stuff with `pencil`...]

# Retrieve and print timing data associated to `plan`
to = get_timer(pencil)
print_timer(to)
```

By default, each `Pencil` has its own `TimerOutput`. If you already have a `TimerOutput`, you can pass it to the [`Pencil`](@ref) constructor:

```julia
to = TimerOutput()
pencil = Pencil(..., timer=to)

# [do stuff with `pencil`...]

print_timer(to)
```

## Library

### Modules

```@docs
Pencils
```

### Types

```@docs
MPITopology
Pencil
```

### Functions

```@docs
get_comm(::MPITopology)
length(::MPITopology)
ndims(::MPITopology)
size(::MPITopology)

eltype(::Pencil)
get_comm(::Pencil)
get_decomposition(::Pencil)
get_permutation(::Pencil)
ndims(::Pencil)
range_local(::Pencil{N}) where N
size_global(::Pencil)
size_local(::Pencil)
to_local(::Pencil)
```

## Index

```@index
Pages = ["Pencils.md"]
Order = [:module, :type, :function]
```

[^1]:
    Why would we want this?
    Perhaps because we want to efficiently perform FFTs along $y$, which under
    this permutation would be the fastest dimension.
