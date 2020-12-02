const ValBool = Union{Val{false}, Val{true}}

# One-dimensional distributed FFT plan.
struct PencilPlan1D{
        Ti <: Number,  # input type
        To <: Number,  # output type
        Pi <: Pencil,
        Po <: Pencil,
        Tr <: AbstractTransform,
        FFTPlanF <: Transforms.Plan,
        FFTPlanB <: Transforms.Plan,
    }
    # Each pencil pair describes the decomposition of input and output FFT
    # data. The two pencils will be different for transforms that do not
    # preserve the size of the data (e.g. real-to-complex transforms).
    # Otherwise, they will be typically identical.
    pencil_in  :: Pi  # pencil before transform
    pencil_out :: Po  # pencil after transform
    transform  :: Tr  # transform type

    fft_plan   :: FFTPlanF  # forward FFTW plan
    bfft_plan  :: FFTPlanB  # backward FFTW plan (unnormalised)

    scale_factor :: Int  # scale factor for backward transform

    function PencilPlan1D{Ti}(p_i, p_o, tr, fw, bw, scale) where {Ti}
        To = eltype_output(tr, Ti)
        new{Ti, To, typeof(p_i), typeof(p_o), typeof(tr), typeof(fw), typeof(bw)}(
            p_i, p_o, tr, fw, bw, scale,
        )
    end
end

Transforms.eltype_input(::PencilPlan1D{Ti}) where {Ti} = Ti
Transforms.eltype_output(::PencilPlan1D{Ti,To}) where {Ti,To} = To
Transforms.is_inplace(p::PencilPlan1D) = is_inplace(p.transform)

"""
    PencilFFTPlan{T,N}

Plan for N-dimensional FFT-based transform on MPI-distributed data.

---

    PencilFFTPlan(
        size_global::Dims{N}, transforms, proc_dims::Dims{M}, comm::MPI.Comm,
        [real_type = Float64];
        extra_dims = (),
        fftw_flags = FFTW.ESTIMATE,
        fftw_timelimit = FFTW.NO_TIMELIMIT,
        permute_dims = Val(true),
        transpose_method = Transpositions.PointToPoint(),
        timer = TimerOutput(),
    )

Create plan for N-dimensional transform.

# Extended help

`size_global` specifies the global dimensions of the input data.

`transforms` should be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[Transform types](@ref). Alternatively, `transforms` may be a
single transform that will be automatically expanded into `N` equivalent
transforms. This is illustrated in the example below.

The transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms. For multidimensional FFTs of
real data, this means that a real-to-complex FFT must be performed along
the first dimension, and then complex-to-complex FFTs are performed
along the other two dimensions (see example below).

The data is distributed over the MPI processes in the `comm` communicator.
The distribution is performed over `M` dimensions (with `M < N`) according to
the values in `proc_dims`, which specifies the number of MPI processes to put
along each dimension.

## Optional arguments

- The floating point precision can be selected by setting `real_type` parameter,
  which is `Float64` by default.

- `extra_dims` may be used to specify the sizes of one or more extra dimensions
  that should not be transformed. These dimensions will be added to the rightmost
  (i.e. slowest) indices of the arrays. See **Extra dimensions** below for usage
  hints.

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
  data transpositions. See
  [PencilArrays docs](https://jipolanco.github.io/PencilArrays.jl/dev/Transpositions/#PencilArrays.Transpositions.Transposition)
  docs for details.

- `timer` should be a `TimerOutput` object.
  See [Measuring performance](@ref PencilFFTs.measuring_performance) for details.

## Extra dimensions

One possible application of `extra_dims` is for describing the components of a
vector or tensor field. However, this means that different `PencilFFTPlan`s
would need to be created for each kind of field (scalar, vector, ...).
To avoid the creation of multiple plans, a possibly better alternative is to
create tuples (or arrays) of `PencilArray`s using [`allocate_input`](@ref) and
[`allocate_output`](@ref).

Another more legitimate usage of `extra_dims` is to specify one or more
Cartesian dimensions that should not be transformed nor split among MPI
processes.

## Example

Suppose we want to perform a 3D FFT of real data. The data is to be
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
struct PencilFFTPlan{
        T <: FFTReal,
        N,   # dimension of arrays (= Nt + Ne)
        I,   # in-place (Bool)
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
    scale_factor :: Float64

    # `method` parameter passed to `transpose!`
    transpose_method :: TransposeMethod

    # Temporary data buffers.
    ibuf :: Vector{UInt8}
    obuf :: Vector{UInt8}

    # Runtime timing.
    # Should be used along with the @timeit_debug macro, to be able to turn it
    # off if desired.
    timer :: TimerOutput

    function PencilFFTPlan(
            A::PencilArray, transforms::AbstractTransformList;
            fftw_flags = FFTW.ESTIMATE,
            fftw_timelimit = FFTW.NO_TIMELIMIT,
            permute_dims::ValBool = Val(true),
            transpose_method::AbstractTransposeMethod =
                Transpositions.PointToPoint(),
            timer::TimerOutput = TimerOutput(),
            ibuf = UInt8[], obuf = UInt8[],  # temporary data buffers
        )
        check_input_array(A, transforms)
        T = real(eltype(A))
        dims_global = size_global(pencil(A), LogicalOrder())
        g = GlobalFFTParams(dims_global, transforms, T)
        inplace = is_inplace(g)
        fftw_kw = (; flags = fftw_flags, timelimit = fftw_timelimit)

        # Options for creation of 1D plans.
        plans = _create_plans(
            A, g;
            permute_dims = permute_dims,
            ibuf = ibuf,
            obuf = obuf,
            timer = timer,
            fftw_kw = fftw_kw,
        )

        scale = prod(p -> float(p.scale_factor), plans)

        # If the plan is in-place, the buffers won't be needed anymore, so we
        # free the memory.
        if inplace
            resize!.((ibuf, obuf), 0)
        end

        edims = extra_dims(A)
        Nt = length(transforms)
        Ne = length(edims)
        N = Nt + Ne
        G = typeof(g)
        P = typeof(plans)
        TM = typeof(transpose_method)
        t = topology(A)
        Nd = ndims(t)

        new{T, N, inplace, Nt, Nd, Ne, G, P, TM}(
            g, t, edims, plans, scale, transpose_method, ibuf, obuf, timer)
    end
end

function PencilFFTPlan(
        dims_global::Dims{Nt}, transforms::AbstractTransformList{Nt},
        proc_dims::Dims{Nd}, comm::MPI.Comm, ::Type{T} = Float64;
        extra_dims::Dims{Ne} = (),
        timer = TimerOutput(),
        ibuf = UInt8[],
        kws...,
    ) where {Nt, Nd, Ne, T <: FFTReal}
    t = MPITopology(comm, proc_dims)
    pen = _make_input_pencil(dims_global, t, timer)
    Ti = eltype_input(first(transforms), T)
    A = _temporary_pencil_array(Ti, pen, ibuf, extra_dims)
    PencilFFTPlan(A, transforms; timer = timer, ibuf = ibuf, kws...)
end

function PencilFFTPlan(A, transform::AbstractTransform,
                       args...; kws...)
    N = _ndims_transformable(A)
    transforms = expand_dims(transform, Val(N))
    PencilFFTPlan(A, transforms, args...; kws...)
end

@inline _ndims_transformable(dims::Dims) = length(dims)
@inline _ndims_transformable(A::PencilArray) = ndims(pencil(A))

"""
    Transforms.is_inplace(p::PencilFFTPlan)

Returns `true` if the given plan operates in-place on the input data, `false`
otherwise.
"""
Transforms.is_inplace(p::PencilFFTPlan{T,N,I}) where {T,N,I} = I :: Bool

"""
    Transforms.eltype_input(p::PencilFFTPlan)

Returns the element type of the input data.
"""
Transforms.eltype_input(p::PencilFFTPlan) = eltype_input(first(p.plans))

"""
    Transforms.eltype_input(p::PencilFFTPlan)

Returns the element type of the output data.
"""
Transforms.eltype_output(p::PencilFFTPlan) = eltype_output(last(p.plans))

pencil_input(p::PencilFFTPlan) = first(p.plans).pencil_in
pencil_output(p::PencilFFTPlan) = last(p.plans).pencil_out

function check_input_array(A::PencilArray, transforms)
    # TODO relax condition to ndims(A) >= N and transform the first N
    # dimensions (and forget about extra_dims)
    N = length(transforms)
    if ndims(pencil(A)) != N
        throw(ArgumentError(
            "number of transforms ($N) must be equal to number " *
            "of transformable dimensions in array (`ndims(pencil(A))`)"
        ))
    end

    if permutation(A) != NoPermutation()
        throw(ArgumentError("dimensions of input array must be unpermuted"))
    end

    decomp = decomposition(pencil(A))  # decomposed dimensions, e.g. (2, 3)
    M = length(decomp)
    decomp_expected = input_decomposition(N, Val(M))
    if decomp != decomp_expected
        throw(ArgumentError(
            "decomposed dimensions of input data must be $decomp_expected" *
            " (got $decomp)"
        ))
    end

    T = eltype(A)
    tr = first(transforms)
    T_expected = eltype_input(tr, real(T))
    if T_expected ∉ (nothing, T)  # if nothing, both real and complex inputs are allowed
        throw(ArgumentError(
            "wrong input datatype ($T) for transform $tr (expected $T_expected)"
        ))
    end

    nothing
end

input_decomposition(N, ::Val{M}) where {M} = ntuple(d -> N - M + d, Val(M))

function _create_plans(A::PencilArray, g::GlobalFFTParams; kws...)
    Tin = input_data_type(g)
    transforms = g.transforms
    opts = (; kws...)  # TODO pass as kwargs
    _create_plans(Tin, g, A, opts, nothing, transforms...)
end

# Create 1D plans recursively.
function _create_plans(
        ::Type{Ti}, g::GlobalFFTParams{T,N} where T,
        Ai::PencilArray, plan1d_opt::NamedTuple,
        plan_prev, transform_fw::AbstractTransform,
        transforms_next::Vararg{AbstractTransform,Ntr}) where {Ti, N, Ntr}
    dim = Val(N - Ntr)  # current dimension index
    n = N - Ntr
    si = g.size_global_in
    so = g.size_global_out
    timer = plan1d_opt.timer
    ibuf = plan1d_opt.ibuf

    Pi = pencil(Ai)

    # Output transform along dimension `n`.
    Po = let dims = ntuple(j -> j ≤ n ? so[j] : si[j], Val(N))
        if dims === size_global(Pi)
            Pi  # in this case Pi and Po are the same
        else
            Pencil(Pi, size_global=dims, timer=timer)
        end
    end

    To = eltype_output(transform_fw, Ti)

    # Note that Ai and Ao may share memory, but that's ok here.
    Ao = _temporary_pencil_array(To, Po, ibuf, extra_dims(Ai))
    plan_n = _make_1d_fft_plan(dim, Ti, Ai, Ao, transform_fw, plan1d_opt)

    # These are both `nothing` when there's no transforms left
    Pi_next = _make_pencil_in(g, topology(Pi), Val(n + 1), plan_n, timer,
                              plan1d_opt.permute_dims)
    Ai_next = _temporary_pencil_array(To, Pi_next, ibuf, extra_dims(Ai))

    (
        plan_n,
        _create_plans(To, g, Ai_next, plan1d_opt, plan_n, transforms_next...)...,
    )
end

# No transforms left!
_create_plans(::Type, ::GlobalFFTParams, ::Nothing, ::NamedTuple, plan_prev) = ()

function _make_input_pencil(dims_global, topology, timer)
    # This is the case of the first pencil pair.
    # Generate initial pencils for the first dimension.
    # - Decompose along dimensions "far" from the first one.
    #   Example: if N = 5 and M = 2, then decomp_dims = (4, 5).
    # - No permutation is applied for input data: arrays are accessed in the
    #   natural order (i1, i2, ..., iN).
    N = length(dims_global)
    M = ndims(topology)
    decomp_dims = input_decomposition(N, Val(M))
    perm = NoPermutation()
    Pencil(topology, dims_global, decomp_dims; permute=perm, timer=timer)
end

function _make_pencil_in(g::GlobalFFTParams,
                         topology::MPITopology, dim::Val{1},
                         plan_prev::Nothing, timer,
                         permute_dims::ValBool)
    _make_input_pencil(g.size_global_in, topology, timer)
end

function _make_pencil_in(g::GlobalFFTParams{T,N} where T,
                         topology::MPITopology{M}, dim::Val{n},
                         plan_prev::PencilPlan1D, timer,
                         permute_dims::ValBool,
                        ) where {N, M, n}
    n > N && return nothing
    Po_prev = plan_prev.pencil_out

    # (i) Determine permutation of pencil data.
    perm = _make_permutation_in(permute_dims, dim, Val(N))

    # (ii) Determine decomposed dimensions from the previous
    # decomposition `n - 1`.
    # If `n` was decomposed previously, shift its associated value
    # in `decomp_prev` to the left.
    # Example: if n = 3 and decomp_prev = (1, 3), then decomp = (1, 2).
    decomp_prev = decomposition(Po_prev)
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
    # (Including dimensions, MPI topology and data buffers.)
    Pencil(Po_prev, decomp_dims=decomp, permute=perm, timer=timer)
end

# No permutations
_make_permutation_in(permute_dims::Val{false}, etc...) = NoPermutation()

function _make_permutation_in(::Val{true}, dim::Val{n}, ::Val{N}) where {n, N}
    # Here the data is permuted so that the n-th logical dimension is the first
    # (fastest) dimension in the arrays.
    # The chosen permutation is equivalent to (n, (1:n-1)..., (n+1:N)...).
    t = ntuple(i -> (i == 1) ? n : (i ≤ n) ? (i - 1) : i, Val(N))
    @assert isperm(t)
    @assert t == (n, (1:n-1)..., (n+1:N)...)
    Permutation(t)
end

# Case n = 1: no permutation of input data
_make_permutation_in(::Val{true}, dim::Val{1}, ::Val) = NoPermutation()

# Case n = N
function _make_permutation_in(::Val{true}, dim::Val{N}, ::Val{N}) where {N}
    # This is the last transform, and I want the index order to be
    # exactly reversed (easier to work with than the alternative above).
    Permutation(ntuple(i -> N - i + 1, Val(N)))  # (N, N-1, ..., 2, 1)
end

function _make_1d_fft_plan(
        dim::Val{n}, ::Type{Ti}, A_fw::PencilArray, A_bw::PencilArray,
        transform_fw::AbstractTransform, plan1d_opt::NamedTuple) where {n, Ti}
    Pi = pencil(A_fw)
    Po = pencil(A_bw)
    perm = permutation(Pi)

    dims = if PencilArrays.isidentity(perm)
        n  # no index permutation
    else
        # Find index of n-th dimension in the permuted array.
        # If we permuted data to have the n-th dimension as the fastest
        # (leftmost) index, then the result of `findfirst` should be 1.
        findfirst(==(n), Tuple(perm)) :: Int
    end

    transform_bw = binv(transform_fw)

    # Scale factor to be applied after backward transform.
    # The passed array must have the dimensions of the backward transform output
    # (i.e. the forward transform input)
    scale_bw = scale_factor(transform_bw, parent(A_fw), dims)

    # Generate forward and backward FFTW transforms.
    fftw_kw = plan1d_opt.fftw_kw
    plan_fw = plan(transform_fw, parent(A_fw), dims; fftw_kw...)

    plan_bw = if transform_bw === Transforms.BRFFT()
        d = size(parent(A_fw), 1)
        plan(transform_bw, parent(A_bw), d, dims; fftw_kw...)
    else
        plan(transform_bw, parent(A_bw), dims; fftw_kw...)
    end

    PencilPlan1D{Ti}(Pi, Po, transform_fw, plan_fw, plan_bw, scale_bw)
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
    scale_factor(p::PencilFFTPlan)

Get scale factor associated to a `PencilFFTPlan`.
"""
scale_factor(p::PencilFFTPlan) = p.scale_factor

"""
    timer(p::PencilFFTPlan)

Get `TimerOutput` attached to a `PencilFFTPlan`.

See [Measuring performance](@ref PencilFFTs.measuring_performance) for details.
"""
timer(p::PencilFFTPlan) = p.timer
