# Available transforms

```@meta
CurrentModule = PencilFFTs.Transforms
```

```@docs
Transforms
```

## Transform types

```@docs
FFT
BFFT

RFFT
BRFFT

R2R

NoTransform
```

## Internals

What follows is used internally by `PencilFFTs`.

### Types

```@docs
AbstractTransform
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

## Index

```@index
Pages = ["Transforms.md"]
Order = [:module, :type, :function]
```
