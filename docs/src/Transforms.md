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
FFT!
BFFT
BFFT!

RFFT
BRFFT

R2R
R2R!

NoTransform
NoTransform!
```

## Internals

What follows is used internally by `PencilFFTs`.

### Types

```@docs
AbstractTransform
IdentityPlan
IdentityPlan!
```

### Functions

```@docs
plan

binv
scale_factor

eltype_input
eltype_output
expand_dims
is_inplace
kind
length_output
```

## Index

```@index
Pages = ["Transforms.md"]
Order = [:module, :type, :function]
```
