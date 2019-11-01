const PencilPair = Tuple{Pencil, Pencil}

"""
    PencilFFTPlan{N,M}

Plan for N-dimensional FFT-based transform on MPI-distributed data.

---

    PencilFFTPlan(size_global::Dims{N}, transforms::AbstractTransformList{N},
                  proc_dims::Dims{M}, comm::MPI.Comm) where {N, M}

Create plan for N-dimensional transform.

`size_global` specifies the global dimensions of the input data.

`transforms` must be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[`Transform types`](@ref Transforms).

The transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms. For multidimensional transforms of
real data, this means that a real-to-complex transform must be performed along
the first dimension, and then complex-to-complex transforms are performed
along the other two dimensions (see example below).

The data is distributed over the MPI processes in the `comm` communicator.
The distribution is performed over `M` dimensions (with `M < N`) according to
the values in `proc_dims`, which specifies the number of MPI processes to put
along each dimension.

# Example

Suppose we want to perform a 3D transform of real data. The data is to be
decomposed along two dimensions, over 8 MPI processes:

```julia
size_global = (64, 32, 128)  # size of real input data

# Perform real-to-complex transform along the first dimension, then
# complex-to-complex transforms along the other dimensions.
transforms = (Transform.RFFT(), Transform.FFT(), Transform.FFT())

proc_dims = (4, 2)  # 2D decomposition
comm = MPI.COMM_WORLD

plan = PencilFFTPlan(size_global, transforms, proc_dims, comm)
```

"""
struct PencilFFTPlan{N,
                     M,
                     G <: GlobalFFTParams}
    global_params :: G
    topology      :: MPITopology{M}

    # Data decomposition configurations.
    # Each pencil pair describes the decomposition of input and output FFT
    # data along the same M dimensions. The two pencils in a pair will
    # be different for transforms that do not preserve the size of the data
    # (e.g. real-to-complex transforms). Otherwise, they will be typically
    # identical.
    pencils :: NTuple{N, PencilPair}

    # TODO
    # - add constructor with Cartesian MPI communicator, in case the user
    #   already created one
    # - allow more control on the decomposition directions
    function PencilFFTPlan(size_global_in::Dims{N},
                           transforms::AbstractTransformList{N},
                           proc_dims::Dims{M}, comm::MPI.Comm) where {N, M}
        global_params = GlobalFFTParams(size_global, transforms)
        topology = MPITopology(comm, proc_dims)
        pencils = _create_pencils(global_params, topology)
        new{N, M, typeof(global_params)}(global_params, topology, pencils)
    end
end

function _create_pencils(global_params::GlobalFFTParams,
                         topology::MPITopology)
    ntuple()
end
