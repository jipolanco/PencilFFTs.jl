const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal
const PlanArrayPair{P,A} = Pair{P,A} where {P <: PencilPlan1D, A <: PencilArray}

# This allows to treat plans as scalars when broadcasting.
# This means that, if u = (u1, u2, u3) is a tuple of PencilArrays
# compatible with p, then p .* u does what one would expect, that is, it
# transforms the three components and returns a tuple.
Broadcast.broadcastable(p::PencilFFTPlan) = Ref(p)

## Forward transforms
function LinearAlgebra.mul!(dst::PencilArray{To,N}, p::PencilFFTPlan{T,N},
                            src::PencilArray{Ti,N}) where {T, N,
                                             Ti <: RealOrComplex{T},
                                             To <: RealOrComplex{T}}
    @timeit_debug p.timer "PencilFFTs mul!" begin
        _check_arrays(p, src, dst)
        _apply_plans!(Val(FFTW.FORWARD), p, dst, src, p.plans...)
    end
end

function Base.:*(p::PencilFFTPlan, src::PencilArray)
    @timeit_debug p.timer "PencilFFTs *" begin
        _check_arrays(p, src, nothing)
        dst = allocate_output(p)
        mul!(dst, p, src)
    end
end

## Backward transforms
function LinearAlgebra.ldiv!(dst::PencilArray{To,N}, p::PencilFFTPlan{T,N},
                             src::PencilArray{Ti,N}) where {T, N,
                                             Ti <: RealOrComplex{T},
                                             To <: RealOrComplex{T}}
    @timeit_debug p.timer "PencilFFTs ldiv!" begin
        _check_arrays(p, dst, src)
        plans = reverse(p.plans)  # plans are applied from right to left

        # TODO can I fuse transform + scaling into one operation? (maybe using
        # callbacks?)
        _apply_plans!(Val(FFTW.BACKWARD), p, dst, src, plans...)

        # Normalise transform
        @timeit_debug p.timer "normalise" ldiv!(p.scale_factor, dst)
    end
end

function Base.:\(p::PencilFFTPlan, src::PencilArray)
    @timeit_debug p.timer "PencilFFTs \\" begin
        _check_arrays(p, nothing, src)
        dst = allocate_input(p)
        ldiv!(dst, p, src)
    end
end

## Operations for collections.
function check_compatible(a::PencilArrayCollection, b::PencilArrayCollection)
    Na = length(a)
    Nb = length(b)
    if Na != Nb
        throw(ArgumentError("collections have different lengths: $Na ≠ $Nb"))
    end
    nothing
end

for f in (:mul!, :ldiv!)
    @eval LinearAlgebra.$f(dst::PencilArrayCollection, p::PencilFFTPlan,
                           src::PencilArrayCollection) =
        (check_compatible(dst, src); $f.(dst, p, src))
end

for f in (:*, :\)
    @eval Base.$f(p::PencilFFTPlan, src::PencilArrayCollection) =
        $f.(p, src)
end

_get_pencils_and_plan(::Val{FFTW.FORWARD}, p::PencilPlan1D) =
    (p.pencil_in, p.pencil_out, p.fft_plan)

_get_pencils_and_plan(::Val{FFTW.BACKWARD}, p::PencilPlan1D) =
    (p.pencil_out, p.pencil_in, p.bfft_plan)

# Out-of-place version
function _apply_plans!(dir::Val, full_plan::PencilFFTPlan,
                       y::PencilArray, x::PencilArray,
                       plan::PencilPlan1D, next_plans::Vararg{PencilPlan1D})
    @assert !is_inplace(full_plan) && !is_inplace(plan)
    Pi, Po, fftw_plan = _get_pencils_and_plan(dir, plan)

    # Transpose data if required.
    u = if pencil(x) === Pi
        x
    else
        u = _temporary_pencil_array(Pi, full_plan.ibuf, full_plan.extra_dims)
        transpose!(u, x, method=full_plan.transpose_method)
    end

    v = if pencil(y) === Po
        y
    else
        _temporary_pencil_array(Po, full_plan.obuf, full_plan.extra_dims)
    end

    @timeit_debug full_plan.timer "FFT" mul!(parent(v), fftw_plan, parent(u))

    _apply_plans!(dir, full_plan, y, v, next_plans...)
end

_apply_plans!(::Val, ::PencilFFTPlan, y::PencilArray, x::PencilArray) = y

# In-place version
function _apply_plans!(dir::Val, full_plan::PencilFFTPlan,
                       A::ManyPencilArray)
    @assert is_inplace(full_plan)
    pairs = _make_pairs(full_plan.plans, A.arrays)

    # Backward transforms are applied in reverse order.
    pp = dir === Val(FFTW.BACKWARD) ? reverse(pairs) : pairs

    _apply_plans!(dir, full_plan, nothing, pp...)

    if dir === Val(FFTW.BACKWARD)
        # Scale transform.
        ldiv!(full_plan.scale_factor, first(A))
    end

    A
end

function _apply_plans!(dir::Val, full_plan::PencilFFTPlan,
                       u_prev::Union{Nothing, PencilArray},
                       pair::PlanArrayPair, next_pairs...)
    plan = pair.first
    u = pair.second
    Pi, Po, fftw_plan = _get_pencils_and_plan(dir, plan)

    @assert is_inplace(full_plan) && is_inplace(plan)
    @assert pencil(u) === plan.pencil_in === plan.pencil_out

    # Buffers should take no memory for in-place transforms.
    @assert length(full_plan.ibuf) == length(full_plan.obuf) == 0

    if u_prev !== nothing
        # Transpose data from previous configuration.
        @assert pointer(u_prev) === pointer(u)  # they're aliased!
        transpose!(u, u_prev, method=full_plan.transpose_method)
    end

    # Perform in-place FFT
    @timeit_debug full_plan.timer "FFT!" fftw_plan * parent(u)

    _apply_plans!(dir, full_plan, u, next_pairs...)
end

_apply_plans!(::Val, ::PencilFFTPlan, u_prev::PencilArray) = u_prev

_split_first(a, b...) = (a, b)  # (x, y, z, w) -> (x, (y, z, w))

function _make_pairs(plans::Tuple{Vararg{PencilPlan1D,N}},
                     arrays::Tuple{Vararg{PencilArray,N}}) where {N}
    p, p_next = _split_first(plans...)
    a, a_next = _split_first(arrays...)
    (p => a, _make_pairs(p_next, a_next)...)
end

_make_pairs(::Tuple{}, ::Tuple{}) = ()

function _check_arrays(p::PencilFFTPlan, xin, xout)
    if xin !== nothing && first(p.plans).pencil_in !== pencil(xin)
        throw(ArgumentError("unexpected dimensions of input data"))
    end
    if xout !== nothing && last(p.plans).pencil_out !== pencil(xout)
        throw(ArgumentError("unexpected dimensions of output data"))
    end
    nothing
end

function _temporary_pencil_array(p::Pencil, buf::Vector{UInt8},
                                 extra_dims::Dims)
    # Create "unsafe" pencil array wrapping buffer data.
    T = eltype(p)
    dims = (size_local(p, permute=true)..., extra_dims...)
    nb = prod(dims) * sizeof(T)
    resize!(buf, nb)
    x = Pencils.unsafe_as_array(T, buf, dims)
    PencilArray(p, x)
end
