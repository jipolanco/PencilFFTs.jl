const FFTWPlanOrIdentity = Union{FFTW.FFTWPlan, Transforms.IdentityPlan}
const ValBool = Union{Val{false}, Val{true}}

# One-dimensional distributed FFT plan.
struct PencilPlan1D{Pi <: Pencil,
                    Po <: Pencil,
                    Tr <: AbstractTransform,
                    FFTPlanF <: FFTWPlanOrIdentity,
                    FFTPlanB <: FFTWPlanOrIdentity,
                   }
    # Each pencil pair describes the decomposition of input and output FFT
    # data. The two pencils will be different for transforms that do not
    # preserve the size and element type of the data (e.g. real-to-complex
    # transforms). Otherwise, they will be typically identical.
    pencil_in  :: Pi       # pencil before transform
    pencil_out :: Po       # pencil after transform
    transform  :: Tr       # transform type

    fft_plan   :: FFTPlanF  # forward FFTW plan
    bfft_plan  :: FFTPlanB  # backward FFTW plan (unnormalised)

    scale_factor :: Int  # scale factor for backward transform
end

"""
    PencilFFTPlan{T,N,M}

Plan for N-dimensional FFT-based transform on MPI-distributed data.

---

    PencilFFTPlan(size_global::Dims{N}, transforms,
                  proc_dims::Dims{M}, comm::MPI.Comm, [real_type=Float64];
                  extra_dims=(),
                  fftw_flags=FFTW.ESTIMATE, fftw_timelimit=FFTW.NO_TIMELIMIT,
                  permute_dims=Val(true),
                  transpose_method=TransposeMethods.IsendIrecv(),
                  timer=TimerOutput(),
                  )

Create plan for N-dimensional transform.

`size_global` specifies the global dimensions of the input data.

`transforms` should be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[Transform types](@ref). Alternatively, `transforms` may be a
single transform that will be automatically expanded into `N` equivalent
transforms. This is illustrated in the example below.

The transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms. For multidimensional transforms of
real data, this means that a real-to-complex transform must be performed along
the first dimension, and then complex-to-complex transforms are performed
along the other two dimensions (see example below).

The data is distributed over the MPI processes in the `comm` communicator.
The distribution is performed over `M` dimensions (with `M < N`) according to
the values in `proc_dims`, which specifies the number of MPI processes to put
along each dimension.

# Optional arguments

- The floating point precision can be selected by setting `real_type` parameter,
  which is `Float64` by default.

- `extra_dims` may be used to specify the sizes of one or more extra dimensions
  that should not be transformed. These dimensions will be added to the rightmost
  (i.e. slowest) indices of the arrays. Extra dimensions can be used to contain,
  for instance, multiple vector field components in a single array.
  See [`Pencils.PencilArray`](@ref) for more details.

- The keyword arguments `fftw_flags` and `fftw_timelimit` are passed to the
  `FFTW` plan creation functions (see [`AbstractFFTs`
  docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft)).

- `permute_dims` determines whether the indices of the output data should be
  reversed. For instance, if the input data has global dimensions
  `(Nx, Ny, Nz)`, then the output of a complex-to-complex FFT would have
  dimensions `(Nz, Ny, Nx)`. This enables FFTs to always be performed along
  the first (i.e. fastest) array dimension, which could lead to performance
  gains. This option is enabled by default. For type inference reasons, it must
  be a value type (`Val(true)` or `Val(false)`).

- `transpose_method` allows to select between implementations of the global
  data transpositions. See the [`transpose!`](@ref) docs for details.

- `timer` should be a `TimerOutput` object.
  See [Measuring performance](@ref Pencils.measuring_performance) for details.

# Example

Suppose we want to perform a 3D transform of real data. The data is to be
decomposed along two dimensions, over 8 MPI processes:

```julia
size_global = (64, 32, 128)  # size of real input data

# Perform real-to-complex transform along the first dimension, then
# complex-to-complex transforms along the other dimensions.
transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
# transforms = Transforms.RFFT()  # this is equivalent to the above line

proc_dims = (4, 2)  # 2D decomposition
comm = MPI.COMM_WORLD

plan = PencilFFTPlan(size_global, transforms, proc_dims, comm)
```

"""
struct PencilFFTPlan{T,
                     N,   # dimension of arrays (= Nt + Ne)
                     Nt,  # number of transformed dimensions
                     Nd,  # number of decomposed dimensions
                     Ne,  # number of extra dimensions
                     G <: GlobalFFTParams,
                     P <: NTuple{Nt, PencilPlan1D},
                     TransposeMethod <: AbstractTransposeMethod,
                    }
    global_params :: G
    topology      :: MPITopology{Nd}

    extra_dims :: Dims{Ne}

    # One-dimensional plans, including data decomposition configurations.
    plans :: P

    # Scale factor to be applied after backwards transforms.
    scale_factor :: Int

    # `method` parameter passed to `Pencils.transpose!`
    transpose_method :: TransposeMethod

    # Temporary data buffers.
    ibuf :: Vector{UInt8}
    obuf :: Vector{UInt8}

    # Runtime timing.
    # Should be used along with the @timeit_debug macro, to be able to turn it
    # off if desired.
    timer :: TimerOutput

    # TODO
    # - add constructor with Cartesian MPI communicator, in case the user
    #   already created one
    # - allow more control on the decomposition directions
    function PencilFFTPlan(size_global::Dims{Nt},
                           transforms::AbstractTransformList{Nt},
                           proc_dims::Dims{Nd}, comm::MPI.Comm,
                           ::Type{T}=Float64;
                           extra_dims::Dims{Ne}=(),
                           fftw_flags=FFTW.ESTIMATE,
                           fftw_timelimit=FFTW.NO_TIMELIMIT,
                           permute_dims::ValBool=Val(true),
                           transpose_method::AbstractTransposeMethod=
                               TransposeMethods.IsendIrecv(),
                           timer::TimerOutput=TimerOutput(),
                           ibuf=UInt8[], obuf=UInt8[],  # temporary data buffers
                          ) where {Nt, Nd, Ne, T <: FFTReal}
        g = GlobalFFTParams(size_global, transforms, T)
        t = MPITopology(comm, proc_dims)

        fftw_kw = (:flags => fftw_flags, :timelimit => fftw_timelimit)

        # Options for creation of 1D plans.
        plan1d_opt = (permute_dims=permute_dims,
                      ibuf=ibuf,
                      obuf=obuf,
                      timer=timer,
                      fftw_kw=fftw_kw,
                      extra_dims=extra_dims,
                     )

        plans = _create_plans(g, t, plan1d_opt)
        scale = prod(p -> p.scale_factor, plans)

        N = Nt + Ne
        G = typeof(g)
        P = typeof(plans)
        TM = typeof(transpose_method)

        new{T, N, Nt, Nd, Ne, G, P, TM}(
            g, t, extra_dims, plans, scale, transpose_method, ibuf, obuf, timer)
    end

    function PencilFFTPlan(size_global::Dims{N},
                           transform::AbstractTransform,
                           args...; kwargs...) where N
        PencilFFTPlan(size_global, expand_dims(transform, Val(N)),
                      args...; kwargs...)
    end
end

function _create_plans(g::GlobalFFTParams{T, N} where T,
                       topology::MPITopology{M},
                       plan1d_opt::NamedTuple) where {N, M}
    Tin = input_data_type(g)
    transforms = g.transforms
    _create_plans(Tin, g, topology, plan1d_opt, nothing, transforms...)
end

# Create 1D plans recursively.
function _create_plans(::Type{Ti},
                       g::GlobalFFTParams{T, N} where T,
                       topology::MPITopology{M},
                       plan1d_opt::NamedTuple,
                       plan_prev,
                       transform_fw::AbstractTransform,
                       transforms_next::Vararg{AbstractTransform, Ntr}
                      ) where {Ti, N, M, Ntr}
    dim = Val(N - Ntr)  # current dimension index
    n = N - Ntr
    si = g.size_global_in
    so = g.size_global_out
    timer = plan1d_opt.timer

    Pi = _make_pencil_in(Ti, g, topology, Val(n), plan_prev, timer,
                         plan1d_opt.permute_dims)

    # Output transform along dimension `n`.
    To = eltype_output(transform_fw, eltype(Pi))
    Po = let dims = ntuple(j -> j ≤ n ? so[j] : si[j], Val(N))
        if dims === size_global(Pi) && To === eltype(Pi)
            Pi  # in this case Pi and Po are the same
        else
            Pencil(Pi, To, size_global=dims, timer=timer)
        end
    end

    plan_n = _make_1d_fft_plan(dim, Pi, Po, transform_fw, plan1d_opt)

    (plan_n, _create_plans(To, g, topology, plan1d_opt, plan_n,
                           transforms_next...)...)
end

# No transforms left!
_create_plans(::Type, ::GlobalFFTParams, ::MPITopology, ::NamedTuple,
              plan_prev) = ()

function _make_pencil_in(::Type{Ti}, g::GlobalFFTParams{T, N} where T,
                         topology::MPITopology{M}, dim::Val{1},
                         plan_prev::Nothing, timer,
                         permute_dims::ValBool) where {Ti, N, M}
    # This is the case of the first pencil pair.
    # Generate initial pencils for the first dimension.
    # - Decompose along dimensions "far" from the first one.
    #   Example: if N = 5 and M = 2, then decomp_dims = (4, 5).
    # - No permutation is applied for input data: arrays are accessed in the
    #   natural order (i1, i2, ..., iN).
    decomp_dims = ntuple(m -> N - M + m, Val(M))
    perm = _make_permutation_in(permute_dims, Val(1), Val(N))
    @assert perm === nothing
    Pencil(topology, g.size_global_in, decomp_dims, Ti, permute=perm,
           timer=timer)
end

function _make_pencil_in(::Type{Ti}, g::GlobalFFTParams{T, N} where T,
                         topology::MPITopology{M}, dim::Val{n},
                         plan_prev::PencilPlan1D, timer,
                         permute_dims::ValBool,
                        ) where {Ti, N, M, n}
    Po_prev = plan_prev.pencil_out

    # (i) Determine permutation of pencil data.
    perm = _make_permutation_in(permute_dims, dim, Val(N))

    # (ii) Determine decomposed dimensions from the previous
    # decomposition `n - 1`.
    # If `n` was decomposed previously, shift its associated value
    # in `decomp_prev` to the left.
    # Example: if n = 3 and decomp_prev = (1, 3), then decomp = (1, 2).
    decomp_prev = get_decomposition(Po_prev)
    decomp = ntuple(Val(M)) do i
        p = decomp_prev[i]
        p == n ? p - 1 : p
    end

    # Note that if `n` was not decomposed previously, then the
    # decomposed dimensions stay the same.
    @assert n ∈ decomp_prev || decomp === decomp_prev

    # If everything is done correctly, there should be no repeated
    # decomposition dimensions.
    @assert allunique(decomp)

    # Create new pencil sharing some information with Po_prev.
    # (Including data type and dimensions, MPI topology and data buffers.)
    Pencil(Po_prev, decomp_dims=decomp, permute=perm, timer=timer)
end

# No permutations
_make_permutation_in(permute_dims::Val{false}, etc...) = nothing

function _make_permutation_in(::Val{true}, dim::Val{n}, ::Val{N}) where {n, N}
    # Here the data is permuted so that the n-th logical dimension is the first
    # (fastest) dimension in the arrays.
    # The chosen permutation is equivalent to (n, (1:n-1)..., (n+1:N)...).
    t = ntuple(i -> (i == 1) ? n : (i ≤ n) ? (i - 1) : i, Val(N))
    @assert isperm(t)
    @assert t == (n, (1:n-1)..., (n+1:N)...)
    Val(t)
end

# Case n = 1: no permutation of input data
_make_permutation_in(::Val{true}, dim::Val{1}, ::Val) = nothing

# Case n = N
function _make_permutation_in(::Val{true}, dim::Val{N}, ::Val{N}) where {N}
    # This is the last transform, and I want the index order to be
    # exactly reversed (easier to work with than the alternative above).
    Val(ntuple(i -> N - i + 1, Val(N)))  # (N, N-1, ..., 2, 1)
end

function _make_1d_fft_plan(dim::Val{n}, Pi::Pencil, Po::Pencil,
                           transform_fw::AbstractTransform,
                           plan1d_opt::NamedTuple) where {n}
    perm = get_permutation(Pi)

    dims = if perm === nothing
        n  # no index permutation
    else
        # Find index of n-th dimension in the permuted array.
        # If we permuted data to have the n-th dimension as the fastest
        # (leftmost) index, then the result of `findfirst` should be 1.
        findfirst(==(n), Pencils.extract(perm)) :: Int
    end

    # Create temporary arrays with the dimensions required for forward and
    # backward transforms.
    transform_bw = binv(transform_fw)
    extra_dims = plan1d_opt.extra_dims
    A_fw = _temporary_pencil_array(Pi, plan1d_opt.ibuf, extra_dims)
    A_bw = _temporary_pencil_array(Po, plan1d_opt.obuf, extra_dims)

    # Scale factor to be applied after backward transform.
    # The passed array must have the dimensions of the backward transform output
    # (i.e. the forward transform input)
    scale_bw = scale_factor(transform_bw, parent(A_fw), dims)

    # Generate forward and backward FFTW transforms.
    fftw_kw = plan1d_opt.fftw_kw
    plan_fw = plan(transform_fw, parent(A_fw), dims; fftw_kw...)
    plan_bw = plan(transform_bw, parent(A_bw), dims; fftw_kw...)

    PencilPlan1D(Pi, Po, transform_fw, plan_fw, plan_bw, scale_bw)
end

function Base.show(io::IO, p::PencilFFTPlan)
    show(io, p.global_params)
    edims = p.extra_dims
    isempty(edims) || println(io, "Extra dimensions: $edims")
    show(io, p.topology)
    nothing
end

"""
    get_comm(p::PencilFFTPlan)

Get MPI communicator associated to a `PencilFFTPlan`.
"""
get_comm(p::PencilFFTPlan) = get_comm(p.topology)

"""
    get_scale_factor(p::PencilFFTPlan) :: Int

Get scale factor associated to a `PencilFFTPlan`.
"""
get_scale_factor(p::PencilFFTPlan) = p.scale_factor :: Int

"""
    get_timer(p::PencilFFTPlan)

Get `TimerOutput` attached to a `PencilFFTPlan`.

See [Measuring performance](@ref PencilFFTs.measuring_performance) for details.
"""
get_timer(p::PencilFFTPlan) = p.timer

"""
    allocate_input(p::PencilFFTPlan)         -> PencilArray
    allocate_input(p::PencilFFTPlan, N)      -> Vector{PencilArray}
    allocate_input(p::PencilFFTPlan, Val(N)) -> NTuple{N, PencilArray}

Allocate uninitialised [`PencilArray`](@ref) that can hold input data for the
given plan.

The second and third forms respectively allocate a vector and a tuple of `N`
`PencilArray`s.
"""
allocate_input(p::PencilFFTPlan) = PencilArray(first(p.plans).pencil_in,
                                               p.extra_dims)
allocate_input(p::PencilFFTPlan, N) = _allocate_many(allocate_input, p, N)

"""
    allocate_output(p::PencilFFTPlan)         -> PencilArray
    allocate_output(p::PencilFFTPlan, N)      -> Vector{PencilArray}
    allocate_output(p::PencilFFTPlan, Val(N)) -> NTuple{N, PencilArray}

Allocate uninitialised [`PencilArray`](@ref) that can hold output data for the
given plan.

The second and third forms respectively allocate a vector and a tuple of `N`
`PencilArray`s.
"""
allocate_output(p::PencilFFTPlan) = PencilArray(last(p.plans).pencil_out,
                                                p.extra_dims)
allocate_output(p::PencilFFTPlan, N) = _allocate_many(allocate_output, p, N)

_allocate_many(allocator::Function, p::PencilFFTPlan, N::Integer) =
    [allocator(p) for n = 1:N]
_allocate_many(allocator::Function, p::PencilFFTPlan, ::Val{N}) where {N} =
    ntuple(n -> allocator(p), Val(N))
