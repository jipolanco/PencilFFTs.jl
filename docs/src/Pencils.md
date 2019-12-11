# [MPI-distributed data](@id Pencils_module)

```@meta
CurrentModule = PencilFFTs.Pencils
```

The distribution of global data among MPI processes is managed by the [`Pencils`](@ref) module.
This module may be used independently of the FFT functionality.

The [`Pencils`](@ref) module defines types that describe [an MPI Cartesian
topology](@ref sec:mpi_topology) and [the decomposition of data over MPI
processes](@ref sec:pencil_configs).
Also, [array wrappers](@ref Array-wrappers) allow to conveniently (and
efficiently) deal with the MPI-decomposed data.

```@docs
Pencils
```

## [MPI topology](@id sec:mpi_topology)

The [`MPITopology`](@ref) type defines the MPI Cartesian topology of the
decomposition.
In other words, it contains information about the number of decomposed
dimensions, and the number of processes in each of these dimensions.

At the lower level, [`MPITopology`](@ref) uses
[`MPI_Cart_create`](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html)
to define a Cartesian MPI communicator.

```@docs
MPITopology
get_comm(::MPITopology)
length(::MPITopology)
ndims(::MPITopology)
size(::MPITopology)
```

## [Pencil configurations](@id sec:pencil_configs)

The [`Pencil`](@ref) type describes a given MPI decomposition configuration of
multidimensional data.

```@docs
Pencil
```

### Functions

```@docs
eltype(::Pencil)
get_comm(::Pencil)
get_decomposition(::Pencil)
get_permutation(::Pencil)
ndims(::Pencil)
range_local(::Pencil{N}) where N
size_global(::Pencil)
size_local(::Pencil)
```

## Array wrappers

**TODO** add description

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

### Internal documentation

```@docs
to_local(::Pencil)
```

## Index

### Modules

```@index
Pages = ["Pencils.md"]
Order = [:module]
```

### Types

```@index
Pages = ["Pencils.md"]
Order = [:type]
```

### Functions

```@index
Pages = ["Pencils.md"]
Order = [:function]
```
