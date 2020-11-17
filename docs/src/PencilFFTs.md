# Distributed FFT plans

Distributed FFTs are implemented in the `PencilFFTs` module, and are built on
top of the [PencilArrays](https://github.com/jipolanco/PencilArrays.jl) package.

```@meta
CurrentModule = PencilFFTs
```

## Creating plans

```@docs
PencilFFTPlan
```

## Allocating data

```@docs
allocate_input
allocate_output
```

## Methods

```@docs
get_comm(::PencilFFTPlan)
scale_factor(::PencilFFTPlan)
timer(::PencilFFTPlan)
is_inplace(::PencilFFTPlan)
```
