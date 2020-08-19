# [Array wrappers](@id PencilArrays_module)

```@meta
CurrentModule = PencilFFTs.PencilArrays
```

The [`PencilArrays`](@ref) module defines types for handling MPI-distributed
data.

The most important types are:

- [`PencilArray`](@ref): array wrapper including MPI decomposition information.
  Takes *local* indices starting at 1, regardless of the location of each MPI
  process on the global topology.

- [`GlobalPencilArray`](@ref): `PencilArray` wrapper that takes *global*
  indices, which generally don't start at 1.
  See also [Global views](@ref).

```@docs
PencilArrays
```

## Construction

An uninitialised `PencilArray` can be constructed from a [`Pencil`](@ref)
instance as
```julia
pencil = Pencil(#= ... =#)
A = PencilArray{Float64}(undef, pencil)
parent(A)  # returns the Array wrapped by `A`
```
This allocates a new `Array` with the local dimensions and data type associated
to the `Pencil`.

One can also construct a `PencilArray` wrapper from an existing
`AbstractArray`, whose dimensions must be compatible with the `Pencil`
configuration.
For instance, the following works:
```julia
dims = size_local(pencil, permute=true)  # dimensions of data array must be permuted!
data = zeros(dims)
A = PencilArray(pencil, data)
```
Note that `data` does not need to be a `Array`, but can be any subtype of
`AbstractArray`.

It is also possible to construct higher dimensional arrays, as in:
```julia
data = zeros(dims..., 3, 2)
A = PencilArray(pencil, data)
```
This will construct a `PencilArray` where the rightmost dimensions (called
*extra dimensions* in the PencilArrays API) will never be split among MPI
processes.

## Dimension permutations

Unlike the wrapped `AbstractArray`, the `PencilArray` wrapper takes
non-permuted indices.
For instance, if the underlying permutation of the `Pencil` is `(2, 3, 1)`,
then `A[i, j, k]` points to the same value as `parent(A)[j, k, i]`.

## Global views

`PencilArray`s are accessed using local indices that start at 1, regardless of
the location of the subdomain associated to the local process on the global
grid.
Sometimes it may be more convenient to use global indices describing the
position of the local process in the domain.
For this, the [`global_view`](@ref) function is provided that generates an
[`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) wrapper taking
global indices.
For more details, see for instance [the gradient example](@ref
gradient_method_global).

## Types

```@docs
PencilArray
GlobalPencilArray
PencilArrayCollection
MaybePencilArrayCollection
ManyPencilArray
```

## Methods

### PencilArray

```@docs
extra_dims(::PencilArray)
get_comm(::MaybePencilArrayCollection)
get_permutation(::MaybePencilArrayCollection)
global_view(::PencilArray)
ndims_extra(::MaybePencilArrayCollection)
ndims_space(::PencilArray)
parent(::PencilArray)
pencil(::PencilArray)
pointer(::PencilArray)
range_local(::MaybePencilArrayCollection)
size(::PencilArray)
size_local(::MaybePencilArrayCollection)
size_global(::MaybePencilArrayCollection)
```

### ManyPencilArray

```@docs
first(::ManyPencilArray)
getindex(::ManyPencilArray)
last(::ManyPencilArray)
length(::ManyPencilArray)
```
