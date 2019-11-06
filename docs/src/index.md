# PencilFFTs.jl

Documentation for PencilFFTs.jl

```@meta
CurrentModule = PencilFFTs
```

```@contents
```

## Types

```@docs
PencilFFTPlan
```

## Functions

```@docs
allocate_input
allocate_output
get_timer
```

## Measuring performance

It is possible to measure the time spent in different sections of the distributed transforms using the [TimerOutput](https://github.com/KristofferC/TimerOutputs.jl) package. This must be enabled by calling `TimerOutputs.enable_debug_timings(PencilFFTs)` after loading this module (see the [TimerOutput docs](https://github.com/KristofferC/TimerOutputs.jl#overhead) for details).

Minimal example:

```julia
using PencilFFTs
using TimerOutput

# Enable timing of PencilFFTs functions
TimerOutputs.enable_debug_timings(PencilFFTs)

plan = PencilFFTPlan(...)

# [do stuff with `plan`...]

# Retrieve and print timing data associated to `plan`
to = get_timer(plan)
print_timer(to)
```

By default, each `PencilFFTPlan` contains its own `TimerOutput`. If you already have a `TimerOutput`, you can pass it to the [`PencilFFTPlan`](@ref) constructor:

```julia
to = TimerOutputs()
plan = PencilFFTPlan(..., timer=to)

# [...]

print_timer(to)
```


## Devdocs

```@docs
GlobalFFTParams
```
