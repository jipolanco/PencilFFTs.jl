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

R2R
```

## Advanced

What follows is used internally by `PencilFFTs`.

### Custom plans

```@docs
IdentityPlan
```

### Functions

```@docs
plan

binv
scale_factor

eltype_input
eltype_output
expand_dims
kind
length_output
```
