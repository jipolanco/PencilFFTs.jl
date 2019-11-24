# Pencils module

```@meta
CurrentModule = PencilFFTs.Pencils
```

```@docs
Pencils
```

## Types

```@docs
MPITopology
Pencil
PencilArray
ShiftedArrayView
```

## Functions

```@docs
eltype
gather
get_comm
get_decomposition
get_permutation
global_view
has_indices
length
ndims
ndims_extra
parent
pencil
range_local
size
size_global
size_local
to_local
transpose!
```

## [Measuring performance](@id Pencils.measuring_performance)

It is possible to measure the time spent in different sections of the MPI data transposition routines using the [TimerOutput](https://github.com/KristofferC/TimerOutputs.jl) package. This has a (very small) performance overhead, so it is disabled by default. To enable time measurements, call `TimerOutputs.enable_debug_timings(PencilFFTs.Pencils)` after loading `PencilFFTs`. For more details see the [TimerOutput docs](https://github.com/KristofferC/TimerOutputs.jl#overhead).

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
