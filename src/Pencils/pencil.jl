"""
    Pencil{N,M,T}

Describes the decomposition of an `N`-dimensional Cartesian geometry among MPI
processes along `M` directions (with `M < N`).

The `Pencil` describes the decomposition of arrays of element type `T`.

---

    Pencil(topology::MPITopology{M}, size_global::Dims{N},
           decomp_dims::Dims{M}, [element_type=Float64];
           permute::Union{Nothing, Val}=nothing,
           timer=TimerOutput())

Define the decomposition of an `N`-dimensional geometry along `M` dimensions.

The dimensions of the geometry are given by `size_global = (N1, N2, ...)`. The
`Pencil` describes the decomposition of an array of dimensions `size_global` and
type `T` across a group of MPI processes.

Data is distributed over the given `M`-dimensional MPI topology (with `M < N`).
The decomposed dimensions are given by `decomp_dims`.

The optional parameter `perm` should be a (compile-time) tuple defining a
permutation of the data indices. Such permutation may be useful for performance
reasons, since it may be preferable (e.g. for FFTs) that the data is contiguous
along the pencil direction.

It is also possible to pass a `TimerOutput` to the constructor. See
[Measuring performance](@ref Pencils.measuring_performance) for details.

# Examples

Decompose a 3D geometry of global dimensions ``N_x × N_y × N_z = 4×8×12`` along
the second (``y``) and third (``z``) dimensions.
```julia
Pencil(topology, (4, 8, 12), (2, 3))             # data is in (x, y, z) order
Pencil(topology, (4, 8, 12), (2, 3), (3, 2, 1))  # data is in (z, y, x) order
```
In the second case, the actual data is stored in `(z, y, x)` order within
each MPI process.

---

    Pencil(p::Pencil{N,M}, [element_type=eltype(p)];
           decomp_dims::Dims{M}=get_decomposition(p),
           size_global::Dims{N}=size_global(p),
           permute::P=get_permutation(p),
           timer::TimerOutput=get_timer(p))

Create new pencil configuration from an existent one.

This constructor enables sharing temporary data buffers between the two pencil
configurations, leading to reduced global memory usage.
"""
struct Pencil{N,  # spatial dimensions
              M,  # MPI topology dimensions (< N)
              T <: Number,  # element type (e.g. Float64, Complex{Float64})
              P,  # optional index permutation (Nothing or Val{permutation})
             }
    # M-dimensional MPI decomposition info (with M < N).
    topology :: MPITopology{M}

    # Global array dimensions (N1, N2, ...).
    # These dimensions are *before* permutation by perm.
    size_global :: Dims{N}

    # Decomposition directions (sorted in increasing order).
    # Example: for x-pencils, this is (2, 3, ..., N).
    decomp_dims :: Dims{M}

    # Part of the array held by every process.
    # These dimensions are *before* permutation by `perm`.
    axes_all :: Array{ArrayRegion{N}, M}

    # Part of the array held by the local process (before permutation).
    axes_local :: ArrayRegion{N}

    # Part of the array held by the local process (after permutation).
    axes_local_perm :: ArrayRegion{N}

    # Optional axes permutation.
    perm :: P

    # Data buffers for transpositions.
    send_buf :: Vector{UInt8}
    recv_buf :: Vector{UInt8}

    # Timing information.
    timer :: TimerOutput

    function Pencil(topology::MPITopology{M}, size_global::Dims{N},
                    decomp_dims::Dims{M}, ::Type{T}=Float64;
                    permute::Union{Nothing, Val}=nothing,
                    send_buf=UInt8[], recv_buf=UInt8[],
                    timer=TimerOutput(),
                   ) where {N, M, T<:Number}
        if !is_valid_permutation(permute)
            # This is almost the same error thrown by `permutedims`.
            throw(ArgumentError("invalid permutation of dimensions: $permute"))
        end
        _check_selected_dimensions(N, decomp_dims)
        decomp_dims = _sort_dimensions(decomp_dims)
        axes_all = get_axes_matrix(decomp_dims, topology.dims, size_global)
        axes_local = axes_all[topology.coords_local...]
        axes_local_perm = permute_indices(axes_local, permute)
        P = typeof(permute)
        new{N,M,T,P}(topology, size_global, decomp_dims, axes_all, axes_local,
                     axes_local_perm, permute, send_buf, recv_buf, timer)
    end

    function Pencil(p::Pencil{N,M}, ::Type{T}=eltype(p);
                    decomp_dims::Dims{M}=get_decomposition(p),
                    size_global::Dims{N}=size_global(p),
                    permute=get_permutation(p),
                    timer::TimerOutput=get_timer(p),
                   ) where {N, M, T<:Number}
        Pencil(p.topology, size_global, decomp_dims, T;
               permute=permute, timer=timer,
               send_buf=p.send_buf, recv_buf=p.recv_buf)
    end
end

# Verify that `dims` is a subselection of dimensions in 1:N.
function _check_selected_dimensions(N, dims::Dims{M}) where M
    if M >= N
        throw(ArgumentError(
            "number of decomposed dimensions `M` must be less than the " *
            "total number of dimensions N = $N (got M = $M)"))
    end
    if !allunique(dims)
        throw(ArgumentError("dimensions may not be repeated. Got $dims."))
    end
    if !all(1 .<= dims .<= N)
        throw(ArgumentError("dimensions must be in 1:$N. Got $dims."))
    end
    nothing
end

function _sort_dimensions(dims::Dims{N}) where N
    s = sort(collect(dims))
    ntuple(n -> s[n], Val(N))  # convert array to tuple
end

function Base.show(io::IO, p::Pencil)
    print(io,
          """
          Decomposition of $(ndims(p))D data
              Data dimensions: $(size_global(p)) [$(eltype(p))]
              Decomposed dimensions: $(get_decomposition(p))
              Data permutation: $(get_permutation(p))""")
end

"""
    eltype(p::Pencil)

Element type associated to the given pencil.
"""
Base.eltype(::Pencil{N, M, T} where {N, M}) where T = T

"""
    get_timer(p::Pencil)

Get `TimerOutput` attached to a `Pencil`.

See [Measuring performance](@ref Pencils.measuring_performance) for details.
"""
get_timer(p::Pencil) = p.timer

"""
    ndims(p::Pencil)

Number of spatial dimensions associated to pencil data.

This corresponds to the total number of dimensions of the space, which includes
the decomposed and non-decomposed dimensions.
"""
Base.ndims(::Pencil{N}) where N = N

"""
    get_comm(p::Pencil)

Get MPI communicator associated to an MPI decomposition scheme.
"""
get_comm(p::Pencil) = get_comm(p.topology)

"""
    get_permutation(p::Pencil)

Get index permutation associated to the given pencil configuration.

Returns `nothing` if there is no associated permutation.
"""
get_permutation(p::Pencil) = p.perm

"""
    get_decomposition(p::Pencil)

Get tuple with decomposed dimensions of the given pencil configuration.
"""
get_decomposition(p::Pencil) = p.decomp_dims

"""
    range_local(p::Pencil; permute=true)

Local data range held by the pencil.

By default the dimensions are permuted to match those of the associated data
arrays.
"""
range_local(p::Pencil{N}; permute::Bool=true) where N =
    (permute ? p.axes_local_perm : p.axes_local) :: ArrayRegion{N}

"""
    size_local(p::Pencil; permute=true)

Local dimensions of the data held by the pencil.

By default the dimensions are permuted to match those of the associated data
arrays.
"""
size_local(p::Pencil; permute::Bool=true) =
    length.(range_local(p, permute=permute))

"""
    size_global(p::Pencil)

Global dimensions of the Cartesian grid associated to the given domain
decomposition.

Unlike `size_local`, the returned dimensions are *not* permuted to match the
dimensions of the local data.
"""
size_global(p::Pencil) = p.size_global

"""
    to_local(p::Pencil, global_inds; permute=true)

Convert non-permuted global indices to local indices.

Indices are permuted by default using the permutation associated to the pencil
configuration `p`.
"""
function to_local(p::Pencil{N}, global_inds::Indices{N};
                  permute=true) where N
    ind = global_inds .- first.(p.axes_local) .+ 1
    permute ? permute_indices(ind, p.perm) : ind
end

function to_local(p::Pencil{N}, global_inds::ArrayRegion{N};
                  permute=true) where N
    ind = map(global_inds, p.axes_local) do rg, rl
        @assert step(rg) == 1
        δ = 1 - first(rl)
        (first(rg) + δ):(last(rg) + δ)
    end :: ArrayRegion{N}
    permute ? permute_indices(ind, p.perm) : ind
end
