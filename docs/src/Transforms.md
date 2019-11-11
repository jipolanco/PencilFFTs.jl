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
BFFT

RFFT
BRFFT
```

## Custom plans

```@docs
IdentityPlan
```

## Functions

The following functions are used internally by `PencilFFTs`.

```@docs
plan

binv
scale_factor

eltype_input
eltype_output
expand_dims
length_output
```
