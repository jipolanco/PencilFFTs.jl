# Transforms module

```@meta
CurrentModule = PencilFFTs.Transforms
```

```@docs
Transforms
```

## Transform types

```@docs
AbstractTransform

NoTransform

FFT
IFFT
BFFT

RFFT
IRFFT
BRFFT
```

## Functions

The following functions are used internally by `PencilFFTs`.

```@docs
eltype_input
eltype_output
inv
length_output
plan
expand_dims
```
