# Distributed FFTs

Distributed FFTs are built on top of the [`PencilArrays`](@ref) and the
[`Transforms`](@ref) modules, and are implemented in the `PencilFFTs` module.

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
get_comm(::PencilFFTPlan)
get_scale_factor(::PencilFFTPlan)
get_timer(::PencilFFTPlan)
is_inplace(::PencilFFTPlan)
```

## [Measuring performance](@id PencilFFTs.measuring_performance)

It is possible to measure the time spent in different sections of the
distributed transforms using the
[TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) package. This has
a (very small) performance overhead, so it is disabled by default. To enable
time measurements, call `TimerOutputs.enable_debug_timings` after loading
`PencilFFTs` (see below for an example). For more details see the [TimerOutputs
docs](https://github.com/KristofferC/TimerOutputs.jl#overhead).

Minimal example:

```julia
using MPI
using PencilFFTs
using TimerOutputs

# Enable timing of `PencilFFTs` functions
TimerOutputs.enable_debug_timings(PencilFFTs)
TimerOutputs.enable_debug_timings(PencilArrays)
TimerOutputs.enable_debug_timings(Transpositions)

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

## Internals

```@docs
GlobalFFTParams
```

## Index

```@index
Pages = ["PencilFFTs.md"]
Order = [:module, :type, :function]
```
