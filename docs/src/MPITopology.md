# [MPI topology](@id sec:mpi_topology)

The [`MPITopology`](@ref) type defines the MPI Cartesian topology of the
decomposition.
In other words, it contains information about the number of decomposed
dimensions, and the number of processes in each of these dimensions.

## Construction

The main `MPITopology` constructor takes a MPI communicator and a tuple
specifying the number of processes in each dimension.
For instance, to distribute 12 MPI processes on a $3 × 4$ grid:
```julia
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
pdims = (3, 4)
topology = MPITopology(comm, pdims)
```

At the lower level, [`MPITopology`](@ref) uses
[`MPI_Cart_create`](https://www.mpich.org/static/docs/latest/www3/MPI_Cart_create.html)
to define a Cartesian MPI communicator.
For more control, one can also create a Cartesian communicator using
`MPI.Cart_create`, and pass that to `MPITopology`:
```julia
comm = MPI.COMM_WORLD
dims = [3, 4]  # note: array, not tuple!
periods = zeros(Int, N)
reorder = false
comm_cart = MPI.Cart_create(comm, dims, periods, reorder)
topology = MPITopology(comm_cart)
```

## Types

```@docs
MPITopology
```

## Methods

```@docs
get_comm(::MPITopology)
length(::MPITopology)
ndims(::MPITopology)
size(::MPITopology)
```
