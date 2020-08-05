# [Pencil configurations](@id sec:pencil_configs)

A *pencil* configuration refers to a given distribution of multidimensional
data among MPI processes.
This information is encoded in the [`Pencil`](@ref) type.

A pencil configuration includes:
- [MPI topology](@ref sec:mpi_topology) information,
- global and local dimensions of the numerical grid,
- subset of decomposed dimensions,
- type of decomposed data (e.g. `Float64`),
- definition of optional permutation of dimensions.

```@docs
Pencils
```

## Construction

The creation of a new [`Pencil`](@ref) requires a [`MPITopology`](@ref), as
well as the global data dimensions and a list of decomposed dimensions.
Optionally, one can also specify the data type (the default is `Float64`) and
a permutation of dimensions.

For instance, to decompose along the first and third dimensions of a complex
3D dataset,
```julia
topology = MPITopology(#= ... =#)
dims_global = (16, 32, 64)
decomp_dims = (1, 3)  # this requires ndims(topology) == 2
pencil = Pencil(topology, dims_global, decomp_dims, Complex{Float64})
```

One may also want to create multiple pencil configurations that differ, for
instance, on the selection of decomposed dimensions.
For this case, a second constructor is available that takes an already existing
`Pencil` instance.
Calling this constructor should be preferred when possible since it allows
sharing memory buffers (used for instance for [global transpositions](@ref
Global-MPI-operations)) and thus reducing memory usage.
The following creates a `Pencil` equivalent to the one above, but with
different decomposed dimensions:
```julia
pencil_x = Pencil(pencil, decomp_dims=(2, 3))
```
See the [`Pencil`](@ref) documentation for more details.

## Dimension permutations

As mentioned above, a `Pencil` may optionally be given information on dimension
permutations.
In this case, the layout of the data arrays in memory is different from the
logical order of dimensions.

To make this clearer, consider the example above where the global data
dimensions are $N_x × N_y × N_z = 16 × 32 × 64$.
In this case, the logical order is $(x, y, z)$.
Now let's say that we want the memory order of the data to be $(y, z, x)$,[^1]
which corresponds to the permutation `(2, 3, 1)`.

Permutations are passed to the `Pencil` constructor via the `permute` keyword
argument.
For performance reasons, dimension permutations are compile-time constants, and
they should be specified using the [`Permutation`](@ref) type defined in
`PencilArrays`.
For instance,
```julia
permutation = Permutation(2, 3, 1)
pencil = Pencil(#= ... =#, permute=permutation)
```
One can also pass [`NoPermutation`](@ref) as a permutation, which disables
permutations (this is the default).

## Types

```@docs
Pencil
Permutation
NoPermutation
```

## Methods

```@docs
eltype(::Pencil)
get_comm(::Pencil)
get_decomposition(::Pencil)
get_permutation(::Pencil)
length(::Pencil)
ndims(::Pencil)
range_local(::Pencil{N}) where N
size_global(::Pencil)
size_local(::Pencil)
to_local(::Pencil)
```

[^1]:
    Why would we want this?
    Perhaps because we want to efficiently perform FFTs along $y$, which, under
    this permutation, would be the fastest dimension.
