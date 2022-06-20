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
    PencilFFTPlan{T,N} <: AbstractFFTs.Plan{T}

Plan for N-dimensional FFT-based transform on MPI-distributed data, where input
data has type `T`.

---

    PencilFFTPlan(p::Pencil, transforms; kwargs...)

Create a `PencilFFTPlan` for distributed arrays following a given
[`Pencil`](https://jipolanco.github.io/PencilArrays.jl/dev/Pencils/#PencilArrays.Pencils.Pencil)
configuration.
See variant below for details on the specification of `transforms` and on
possible keyword arguments.

---

    PencilFFTPlan(
        A::PencilArray, transforms;
        fftw_flags = FFTW.ESTIMATE,
        fftw_timelimit = FFTW.NO_TIMELIMIT,
        permute_dims = Val(true),
        transpose_method = Transpositions.PointToPoint(),
        timer = TimerOutput(),
    )

Create plan for `N`-dimensional transform on MPI-distributed `PencilArray`s.

# Extended help

This creates a `PencilFFTPlan` for arrays sharing the same properties as `A`
(dimensions, MPI decomposition, memory layout, ...), which describe data on an
`N`-dimensional domain.

## Transforms

The transforms to be applied along each dimension are specified by the
`transforms` argument. Possible transforms are defined as subtypes of
[`Transforms.AbstractTransform`](@ref), and are listed in [Transform
types](@ref). This argument may be either:

- a tuple of `N` transforms to be applied along each dimension. For instance,
  `transforms = (Transforms.R2R(FFTW.REDFT01), Transforms.RFFT(), Transforms.FFT())`;

- a single transform to be applied along all dimensions. The input is
  automatically expanded into `N` equivalent transforms. For instance, for a
  three-dimensional array, `transforms = Transforms.RFFT()` specifies a 3D
  real-to-complex transform, and is equivalent to passing `(Transforms.RFFT(),
  Transforms.FFT(), Transforms.FFT())`.

Note that forward transforms are applied from left to right. In the last
example, this means that a real-to-complex transform (`RFFT`) is first performed along
the first dimension. This is followed by complex-to-complex transforms (`FFT`)
along the second and third dimensions.

## Input data layout

The input `PencilArray` must satisfy the following constraints:

- array dimensions must *not* be permuted. This is the default when constructing
  `PencilArray`s.

- for an `M`-dimensional domain decomposition (with `M < N`), the input array
  must be decomposed along the *last `M` dimensions*. For example, for a 2D
  decomposition of 3D data, the decomposed dimensions must be `(2, 3)`. In
  particular, the first array dimension must *not* be distributed among
  different MPI processes.

  In the PencilArrays package, the decomposed dimensions are specified
  at the moment of constructing a [`Pencil`](https://jipolanco.github.io/PencilArrays.jl/dev/Pencils/#PencilArrays.Pencils.Pencil).

- the element type must be compatible with the specified transform. For
  instance, real-to-complex transforms (`Transforms.RFFT`) require the input to
  be real floating point values. Other transforms, such as `Transforms.R2R`,
  accept both real and complex data.

## Keyword arguments

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

---

    PencilFFTPlan(
        dims_global::Dims{N}, transforms, proc_dims::Dims{M}, comm::MPI.Comm,
        [real_type = Float64]; extra_dims = (), kws...
    )

Create plan for N-dimensional transform.

# Extended help

Instead of taking a `PencilArray` or a `Pencil`, this constructor requires the
global dimensions of the input data, passed via the `size_global` argument.

The data is distributed over the MPI processes in the `comm` communicator.
The distribution is performed over `M` dimensions (with `M < N`) according to
the values in `proc_dims`, which specifies the number of MPI processes to put
along each dimension.

`PencilArray`s that may be transformed with the returned plan can be created
using [`allocate_input`](@ref).

## Optional arguments

- The floating point precision can be selected by setting `real_type` parameter,
  which is `Float64` by default.

- `extra_dims` may be used to specify the sizes of one or more extra dimensions
  that should not be transformed. These dimensions will be added to the rightmost
  (i.e. slowest) indices of the arrays. See **Extra dimensions** below for usage
  hints.

- see the other constructor for more keyword arguments.

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
        T,   # element type of input data
        N,   # dimension of arrays (= Nt + Ne)
        I,   # in-place (Bool)
        Nt,  # number of transformed dimensions
        Nd,  # number of decomposed dimensions
        Ne,  # number of extra dimensions
        G <: GlobalFFTParams,
        P <: NTuple{Nt, PencilPlan1D},
        TransposeMethod <: AbstractTransposeMethod,
        Buffer <: DenseVector{UInt8},
    } <: AbstractFFTs.Plan{T}

    global_params :: G
    topology      :: MPITopology{Nd}

    extra_dims :: Dims{Ne}

    # One-dimensional plans, including data decomposition configurations.
    plans :: P

    # Scale factor to be applied after backwards transforms.
    scale_factor :: Float64

    # `method` parameter passed to `transpose!`
    transpose_method :: TransposeMethod

    # TODO can I reuse the Pencil buffers (send_buf, recv_buf) to reduce allocations?
    # Temporary data buffers.
    ibuf :: Buffer
    obuf :: Buffer

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
            ibuf = _make_fft_buffer(A), obuf = _make_fft_buffer(A),
        )
        T = eltype(A)
        pen = pencil(A)
        dims_global = size_global(pen, LogicalOrder())
        g = GlobalFFTParams(dims_global, transforms, real(T))
        check_input_array(A, g)
        inplace = is_inplace(g)
        fftw_kw = (; flags = fftw_flags, timelimit = fftw_timelimit)

        # Options for creation of 1D plans.
        plans = _create_plans(
            A, g;
            permute_dims = permute_dims,
            ibuf = ibuf,
            timer = timer,
            fftw_kw = fftw_kw,
        )

        scale = prod(p -> float(p.scale_factor), plans)

        # If the plan is in-place, the buffers won't be needed anymore, so we
        # free the memory.
        # TODO this assumes that buffers are not shared with the Pencil object!
        if inplace
            @assert ibuf ∉ (pen.send_buf, pen.recv_buf)
            @assert obuf ∉ (pen.send_buf, pen.recv_buf)
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
        Buffer = typeof(ibuf)

        new{T, N, inplace, Nt, Nd, Ne, G, P, TM, Buffer}(
            g, t, edims, plans, scale, transpose_method, ibuf, obuf, timer,
        )
    end
end

function PencilFFTPlan(
        pen::Pencil{Nt}, transforms::AbstractTransformList{Nt}, ::Type{Tr} = Float64;
        extra_dims::Dims = (), timer = TimerOutput(), ibuf = _make_fft_buffer(pen),
        kws...,
    ) where {Nt, Tr <: FFTReal}
    T = _input_data_type(Tr, transforms...)
    A = _temporary_pencil_array(T, pen, ibuf, extra_dims)
    PencilFFTPlan(A, transforms; timer = timer, ibuf = ibuf, kws...)
end

function PencilFFTPlan(
        dims_global::Dims{Nt}, transforms::AbstractTransformList{Nt},
        proc_dims::Dims, comm::MPI.Comm, ::Type{Tr} = Float64;
        timer = TimerOutput(), kws...,
    ) where {Nt, Tr}
    t = MPITopology(comm, proc_dims)
    pen = _make_input_pencil(dims_global, t, timer)
    PencilFFTPlan(pen, transforms, Tr; timer = timer, kws...)
end

function PencilFFTPlan(A, transform::AbstractTransform, args...; kws...)
    N = _ndims_transformable(A)
    transforms = expand_dims(transform, Val(N))
    PencilFFTPlan(A, transforms, args...; kws...)
end

_make_fft_buffer(p::Pencil) = similar(p.send_buf, UInt8, 0) :: DenseVector{UInt8}
_make_fft_buffer(A::PencilArray) = _make_fft_buffer(pencil(A))

@inline _ndims_transformable(dims::Dims) = length(dims)
@inline _ndims_transformable(p::Pencil) = ndims(p)
@inline _ndims_transformable(A::PencilArray) = _ndims_transformable(pencil(A))

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
    Transforms.eltype_output(p::PencilFFTPlan)

Returns the element type of the output data.
"""
Transforms.eltype_output(p::PencilFFTPlan) = eltype_output(last(p.plans))

pencil_input(p::PencilFFTPlan) = first(p.plans).pencil_in
pencil_output(p::PencilFFTPlan) = last(p.plans).pencil_out

function check_input_array(A::PencilArray, g::GlobalFFTParams)
    # TODO relax condition to ndims(A) >= N and transform the first N
    # dimensions (and forget about extra_dims)
    N = ndims(g)
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
    T_expected = input_data_type(g)
    if T_expected !== T
        throw(ArgumentError("wrong input datatype $T, expected $T_expected\n$g"))
    end

    nothing
end

input_decomposition(N, ::Val{M}) where {M} = ntuple(d -> N - M + d, Val(M))

function _create_plans(A::PencilArray, g::GlobalFFTParams; kws...)
    Tin = input_data_type(g)
    transforms = g.transforms
    _create_plans(Tin, g, A, nothing, transforms...; kws...)
end

# Create 1D plans recursively.
function _create_plans(
        ::Type{Ti}, g::GlobalFFTParams{T,N} where T,
        Ai::PencilArray, plan_prev, transform_fw::AbstractTransform,
        transforms_next::Vararg{AbstractTransform,Ntr};
        timer, ibuf, fftw_kw, permute_dims,
    ) where {Ti, N, Ntr}
    dim = Val(N - Ntr)  # current dimension index
    n = N - Ntr
    si = g.size_global_in
    so = g.size_global_out

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
    plan_n = _make_1d_fft_plan(dim, Ti, Ai, Ao, transform_fw; fftw_kw = fftw_kw)

    # These are both `nothing` when there's no transforms left
    Pi_next = _make_intermediate_pencil(
        g, topology(Pi), Val(n + 1), plan_n, timer, permute_dims)
    Ai_next = _temporary_pencil_array(To, Pi_next, ibuf, extra_dims(Ai))

    (
        plan_n,
        _create_plans(
            To, g, Ai_next, plan_n, transforms_next...;
            timer = timer, ibuf = ibuf, fftw_kw = fftw_kw, permute_dims = permute_dims,
        )...,
    )
end

# No transforms left!
_create_plans(::Type, ::GlobalFFTParams, ::Nothing, plan_prev; kws...) = ()

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

function _make_intermediate_pencil(
        g::GlobalFFTParams{T,N} where T,
        topology::MPITopology{M}, dim::Val{n},
        plan_prev::PencilPlan1D, timer,
        permute_dims::ValBool,
    ) where {N, M, n}
    @assert n ≥ 2  # this is an intermediate pencil
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
    @assert n ≥ 2
    # Here the data is permuted so that the n-th logical dimension is the first
    # (fastest) dimension in the arrays.
    # The chosen permutation is equivalent to (n, (1:n-1)..., (n+1:N)...).
    t = ntuple(i -> (i == 1) ? n : (i ≤ n) ? (i - 1) : i, Val(N))
    @assert isperm(t)
    @assert t == (n, (1:n-1)..., (n+1:N)...)
    Permutation(t)
end

# Case n = N
function _make_permutation_in(::Val{true}, dim::Val{N}, ::Val{N}) where {N}
    # This is the last transform, and I want the index order to be
    # exactly reversed (easier to work with than the alternative above).
    Permutation(ntuple(i -> N - i + 1, Val(N)))  # (N, N-1, ..., 2, 1)
end

function _make_1d_fft_plan(
        dim::Val{n}, ::Type{Ti}, A_fw::PencilArray, A_bw::PencilArray,
        transform_fw::AbstractTransform; fftw_kw) where {n, Ti}
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

    d = size(parent(A_fw), 1)  # input length along transformed dimension
    transform_bw = binv(transform_fw, d)

    # Scale factor to be applied after backward transform.
    # The passed array must have the dimensions of the backward transform output
    # (i.e. the forward transform input)
    scale_bw = scale_factor(transform_bw, parent(A_bw), dims)

    # Generate forward and backward FFTW transforms.
    plan_fw = plan(transform_fw, parent(A_fw), dims; fftw_kw...)
    plan_bw = plan(transform_bw, parent(A_bw), dims; fftw_kw...)

    PencilPlan1D{Ti}(Pi, Po, transform_fw, plan_fw, plan_bw, scale_bw)
end

function Base.show(io::IO, p::PencilFFTPlan)
    show(io, p.global_params)
    edims = extra_dims(p)
    isempty(edims) || println(io, "Extra dimensions: $edims")
    println(io)
    show(io, p.topology)
    nothing
end

"""
    get_comm(p::PencilFFTPlan)

Get MPI communicator associated to a `PencilFFTPlan`.
"""
get_comm(p::PencilFFTPlan) = get_comm(topology(p))

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

# For consistency with AbstractFFTs, this gives the global dimensions of the input.
Base.size(p::PencilFFTPlan) = size_global(pencil_input(p), LogicalOrder())

topology(p::PencilFFTPlan) = p.topology
extra_dims(p::PencilFFTPlan) = p.extra_dims
