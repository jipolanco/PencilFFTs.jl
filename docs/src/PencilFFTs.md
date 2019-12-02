# PencilFFTs module

```@meta
CurrentModule = PencilFFTs
```

## Types

```@docs
PencilFFTPlan
```

## Functions

```@docs
allocate_input
allocate_output
get_scale_factor
get_timer
```

## [Measuring performance](@id PencilFFTs.measuring_performance)

It is possible to measure the time spent in different sections of the distributed transforms using the [TimerOutput](https://github.com/KristofferC/TimerOutputs.jl) package. This has a (very small) performance overhead, so it is disabled by default. To enable time measurements, call `TimerOutputs.enable_debug_timings(PencilFFTs)` and `TimerOutputs.enable_debug_timings(PencilFFTs.Pencils)` after loading `PencilFFTs`. For more details see the [TimerOutput docs](https://github.com/KristofferC/TimerOutputs.jl#overhead).

Minimal example:

```julia
using MPI
using PencilFFTs
using TimerOutput

# Enable timing of `PencilFFTs` functions
TimerOutputs.enable_debug_timings(PencilFFTs)
TimerOutputs.enable_debug_timings(PencilFFTs.Pencils)

MPI.Init()

plan = PencilFFTPlan(#= args... =#)

# [do stuff with `plan`...]

# Retrieve and print timing data associated to `plan`
to = get_timer(plan)
print_timer(to)
```

By default, each `PencilFFTPlan` has its own `TimerOutput`. If you already have a `TimerOutput`, you can pass it to the [`PencilFFTPlan`](@ref) constructor:

```julia
to = TimerOutput()
plan = PencilFFTPlan(..., timer=to)

# [do stuff with `plan`...]

print_timer(to)
```

## Devdocs

```@docs
GlobalFFTParams
```
