const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal

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
        _apply_plans!(Val(FFTW.FORWARD), dst, src, p, p.plans...)
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
        _apply_plans!(Val(FFTW.BACKWARD), dst, src, p, plans...)

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

function _apply_plans!(dir::Val, y::PencilArray, x::PencilArray,
                       full_plan::PencilFFTPlan,
                       plan::PencilPlan1D, next_plans::Vararg{PencilPlan1D})
    if dir === Val(FFTW.FORWARD)
        Pi = plan.pencil_in
        Po = plan.pencil_out
        fftw_plan = plan.fft_plan
    elseif dir === Val(FFTW.BACKWARD)
        Pi = plan.pencil_out
        Po = plan.pencil_in
        fftw_plan = plan.bfft_plan
    end

    # Transpose pencil if required.
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

    @debug("Apply 1D plan", fftw_plan, get_permutation(Pi), size(parent(u)),
           size(parent(v)))

    @timeit_debug full_plan.timer "FFT" mul!(parent(v), fftw_plan, parent(u))

    _apply_plans!(dir, y, v, full_plan, next_plans...)
end

_apply_plans!(::Val, y::PencilArray, x::PencilArray, ::PencilFFTPlan) = y

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
