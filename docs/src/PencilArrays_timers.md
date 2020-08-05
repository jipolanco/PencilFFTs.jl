# [Measuring performance](@id PencilArrays.measuring_performance)

It is possible to measure the time spent in different sections of the MPI data
transposition routines using the
[TimerOutputs](https://github.com/KristofferC/TimerOutputs.jl) package. This
has a (very small) performance overhead, so it is disabled by default. To
enable time measurements, call
`TimerOutputs.enable_debug_timings` after loading `PencilFFTs` (see below for
an example).
For more details see the [TimerOutputs
docs](https://github.com/KristofferC/TimerOutputs.jl#overhead).

Minimal example:

```julia
using MPI
using PencilFFTs.PencilArrays
using TimerOutputs

# Enable timing of `PencilArrays` functions
TimerOutputs.enable_debug_timings(PencilArrays)
TimerOutputs.enable_debug_timings(Transpositions)

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
