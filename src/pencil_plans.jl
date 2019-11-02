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
transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())

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
    # TODO Maybe this should be a tuple of M + 1 pairs.
    # This seems to be the minimal number of configurations required.
    # In the case of slab decomposition in 3D, this would avoid a
    # data transposition!
    # Alternative: identical pencils, with no effective transposition (and no
    # permutation either?).
    pencils :: NTuple{N, PencilPair}

    # TODO
    # - add constructor with Cartesian MPI communicator, in case the user
    #   already created one
    # - allow more control on the decomposition directions
    function PencilFFTPlan(size_global::Dims{N},
                           transforms::AbstractTransformList{N},
                           proc_dims::Dims{M}, comm::MPI.Comm) where {N, M}
        global_params = GlobalFFTParams(size_global, transforms)
        topology = MPITopology(comm, proc_dims)
        pencils = _create_pencils(global_params, topology)
        new{N, M, typeof(global_params)}(global_params, topology, pencils)
    end
end

function _create_pencils(global_params::GlobalFFTParams{N},
                         topology::MPITopology{M}) where {N, M}
    si = global_params.size_global_in
    so = global_params.size_global_out


    # 1. Generate initial pencil for input data.
    # - Decompose along dimensions "far" from the first one.
    #   Example: if N = 5 and M = 2, then decomp_dims = (4, 5).
    # - No permutation is applied for input data: arrays accessed in the natural
    #   order (i1, i2, ..., iN).
    pencil_in = let decomp_dims = ntuple(m -> N - M + m, Val(M))
        Pencil(topology, si, decomp_dims, permute=nothing)
    end

    decomp_prev = get_decomposition(pencil_in)

    # 2. Iterate over dimensions 1:N, generating one pencil pair at a time.
    pens = ntuple(Val(N)) do n
        # Dimensions of input and output pencils.
        dims_i = ntuple(j -> j ≥ n ? si[j] : so[j], Val(N))
        dims_o = ntuple(j -> j > n ? si[j] : so[j], Val(N))

        # Input pencil
        Pi = if n == 1
            @assert size_global(pencil_in) === dims_i
            pencil_in
        else
            # (i) Determine permutation of pencil data.
            # The data is permuted so that the n-th logical dimension is the
            # first (fastest) dimension in the arrays.
            # The chosen permutation is equivalent to (n, (1:n-1)..., (n+1:N)...)
            perm = ntuple(i -> (i == 1) ? n : (i ≤ n) ? (i - 1) : i, Val(N))
            @assert isperm(perm)
            @assert perm == (n, (1:n-1)..., (n+1:N)...)

            # (ii) Determine decomposed dimensions from the previous
            # decomposition `n - 1`.
            # If `n` was decomposed previously, shift its associated value
            # in `decomp_prev` to the left.
            # Example: if n = 3 and decomp_prev = (1, 3), then decomp = (1, 2).
            decomp = ntuple(Val(M)) do i
                p = decomp_prev[i]
                p == n ? p - 1 : p
            end

            # Note that if `n` was not decomposed previously, then the
            # decomposed dimensions stay the same.
            @assert n ∈ decomp_prev || decomp === decomp_prev

            # If everything is done correctly, there should be no repeated
            # decomposition dimensions.
            @assert all(count(decomp .== i) <= 1 for i in 1:N)

            decomp_prev = decomp

            # Create new pencil sharing some information with pencil_in.
            # (Including MPI topology and data buffers.)
            Pencil(pencil_in, decomp, dims_i, permute=perm)
        end

        @debug "PencilFFTPlan: create_pencils" n size_global(Pi) get_decomposition(Pi) get_permutation(Pi)

        # Output pencil.
        # If the input and output dimensions are not the same, generate a
        # similar pencil to Pi but with different dimensions.
        Po = dims_i === dims_o ? Pi : Pencil(Pi, get_decomposition(Pi), dims_o)

        Pi, Po
    end

    pens
end
