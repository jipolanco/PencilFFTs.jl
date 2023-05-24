const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal
const PlanArrayPair{P,A} = Pair{P,A} where {P <: PencilPlan1D, A <: PencilArray}

# Types of array over which a PencilFFTPlan can operate.
# PencilArray, ManyPencilArray and ManyPencilArrayRFFT! are respectively for out-of-place, in-place and in-place rfft
# transforms.
const FFTArray{T,N} = Union{PencilArray{T,N}, ManyPencilArray{T,N}, ManyPencilArrayRFFT!{T,N}} where {T,N}

# Collections of FFTArray (e.g. for vector components), for broadcasting plans
# to each array. These types are basically those returned by `allocate_input`
# and `allocate_output` when optional arguments are passed.
const FFTArrayCollection =
    Union{Tuple{Vararg{A}}, AbstractArray{A}} where {A <: FFTArray}

const PencilMultiarray{T,N} = Union{ManyPencilArray{T,N}, ManyPencilArrayRFFT!{T,N}} where {T,N}

# This allows to treat plans as scalars when broadcasting.
# This means that, if u = (u1, u2, u3) is a tuple of PencilArrays
# compatible with p, then p .* u does what one would expect, that is, it
# transforms the three components and returns a tuple.
Broadcast.broadcastable(p::PencilFFTPlan) = Ref(p)

# Forward transforms
function LinearAlgebra.mul!(
        dst::FFTArray{To,N}, p::PencilFFTPlan{T,N}, src::FFTArray{T,N},
    ) where {T, N, To <: RealOrComplex}
    @timeit_debug p.timer "PencilFFTs mul!" begin
        _check_arrays(p, src, dst)
        _apply_plans!(Val(FFTW.FORWARD), p, dst, src)
    end
end

# Backward transforms (unscaled)
function bmul!(
        dst::FFTArray{T,N}, p::PencilFFTPlan{T,N}, src::FFTArray{Ti,N},
    ) where {T, N, Ti <: RealOrComplex}
    @timeit_debug p.timer "PencilFFTs bmul!" begin
        _check_arrays(p, dst, src)
        _apply_plans!(Val(FFTW.BACKWARD), p, dst, src)
    end
end

# Inverse transforms (scaled)
function LinearAlgebra.ldiv!(
        dst::FFTArray{T,N}, p::PencilFFTPlan{T,N}, src::FFTArray{Ti,N},
    ) where {T, N, Ti <: RealOrComplex}
    @timeit_debug p.timer "PencilFFTs ldiv!" begin
        _check_arrays(p, dst, src)
        _apply_plans!(Val(FFTW.BACKWARD), p, dst, src)
        _scale!(dst, inv(scale_factor(p)))
    end
end

function Base.:*(p::PencilFFTPlan, src::FFTArray)
    dst = _maybe_allocate(allocate_output, p, src)
    mul!(dst, p, src)
end

function Base.:\(p::PencilFFTPlan, src::FFTArray)
    dst = _maybe_allocate(allocate_input, p, src)
    ldiv!(dst, p, src)
end

function _scale!(dst::PencilArray{<:RealOrComplex{T},N}, inv_scale::T) where {T,N}
    parent(dst) .*= inv_scale
end

function _scale!(dst::PencilMultiarray{<:RealOrComplex{T},N}, inv_scale::T) where {T,N}
    parent(first(dst)) .*= inv_scale
end

# Out-of-place version
_maybe_allocate(allocator::Function, p::PencilFFTPlan{T,N,false} where {T,N},
                ::PencilArray) = allocator(p)

# In-place version
_maybe_allocate(::Function, ::PencilFFTPlan{T,N,true} where {T,N},
                src::PencilMultiarray) = src

# Fallback case.
function _maybe_allocate(::Function, p::PencilFFTPlan, src::A) where {A}
    s = is_inplace(p) ? "in-place" : "out-of-place"
    throw(ArgumentError(
        "input array type $A incompatible with $s plans"))
end

function _check_arrays(
        p::PencilFFTPlan{T,N,false} where {T,N},
        Ain::PencilArray, Aout::PencilArray,
    )
    if Base.mightalias(Ain, Aout)
        throw(ArgumentError("out-of-place plan applied to aliased data"))
    end
    _check_pencils(p, Ain, Aout)
    nothing
end

function _check_arrays(
        p::PencilFFTPlan{T,N,true} where {T,N},
        Ain::PencilMultiarray, Aout::PencilMultiarray,
    )
    if Ain !== Aout
        throw(ArgumentError(
            "input and output arrays for in-place plan must be the same"))
    end
    _check_pencils(p, first(Ain), last(Ain))
    nothing
end

# Fallback case: plan type is incompatible with array types.
# For instance, plan is in-place, and at least one of the arrays is a regular
# PencilArray (instead of a ManyPencilArray).
function _check_arrays(p::PencilFFTPlan, ::Ai, ::Ao) where {Ai, Ao}
    s = is_inplace(p) ? "in-place" : "out-of-place"
    throw(ArgumentError(
        "array types ($Ai, $Ao) incompatible with $s plans"))
end

function _check_pencils(p::PencilFFTPlan, Ain::PencilArray, Aout::PencilArray)
    if first(p.plans).pencil_in !== pencil(Ain)
        throw(ArgumentError("unexpected dimensions of input data"))
    end
    if last(p.plans).pencil_out !== pencil(Aout)
        throw(ArgumentError("unexpected dimensions of output data"))
    end
    nothing
end

# Operations for collections.
function check_compatible(a::FFTArrayCollection, b::FFTArrayCollection)
    Na = length(a)
    Nb = length(b)
    if Na != Nb
        throw(ArgumentError("collections have different lengths: $Na ≠ $Nb"))
    end
    nothing
end

for f in (:mul!, :ldiv!)
    @eval LinearAlgebra.$f(dst::FFTArrayCollection, p::PencilFFTPlan,
                           src::FFTArrayCollection) =
        (check_compatible(dst, src); $f.(dst, p, src))
end

bmul!(dst::FFTArrayCollection, p::PencilFFTPlan,
                           src::FFTArrayCollection) =
        (check_compatible(dst, src); bmul!.(dst, p, src))

for f in (:*, :\)
    @eval Base.$f(p::PencilFFTPlan, src::FFTArrayCollection) =
        $f.(p, src)
end

@inline transform_info(::Val{FFTW.FORWARD}, p::PencilPlan1D{Ti,To}) where {Ti,To} =
    (Ti = Ti, To = To, Pi = p.pencil_in, Po = p.pencil_out, fftw_plan = p.fft_plan)

@inline transform_info(::Val{FFTW.BACKWARD}, p::PencilPlan1D{Ti,To}) where {Ti,To} =
    (Ti = To, To = Ti, Pi = p.pencil_out, Po = p.pencil_in, fftw_plan = p.bfft_plan)

# Out-of-place version
function _apply_plans!(
        dir::Val, full_plan::PencilFFTPlan{T,N,false} where {T,N},
        y::PencilArray, x::PencilArray)
    plans = let p = full_plan.plans
        # Backward transforms are applied in reverse order.
        dir === Val(FFTW.BACKWARD) ? reverse(p) : p
    end

    _apply_plans_out_of_place!(dir, full_plan, y, x, plans...)

    y
end

# In-place version
function _apply_plans!(
        dir::Val, full_plan::PencilFFTPlan{T,N,true} where {T,N},
        A::ManyPencilArray, A_again::ManyPencilArray)
    @assert A === A_again
    pairs = _make_pairs(full_plan.plans, A.arrays)

    # Backward transforms are applied in reverse order.
    pp = dir === Val(FFTW.BACKWARD) ? reverse(pairs) : pairs

    _apply_plans_in_place!(dir, full_plan, nothing, pp...)

    A
end

# In-place RFFT version
function _apply_plans!(
    dir::Val, full_plan::PencilFFTPlan{T,N,true},
    A::ManyPencilArrayRFFT!{T,N}, A_again::ManyPencilArrayRFFT!{T,N}) where {T<:FFTReal,N}
    @assert A === A_again
    # pairs for 1D FFT! plans, RFFT! plan is treated separately
    pairs = _make_pairs(full_plan.plans[2:end], A.arrays[3:end])

    # Backward transforms are applied in reverse order.
    pp = dir === Val(FFTW.BACKWARD) ? reverse(pairs) : pairs

    if dir === Val(FFTW.FORWARD)
        # apply separately first transform (RFFT!)
        _apply_rfft_plan_in_place!(dir, full_plan, A.arrays[2], first(full_plan.plans), A.arrays[1])
        # apply recursively all successive transforms (FFT!)
        _apply_plans_in_place!(dir, full_plan, A.arrays[2], pp...)
    elseif dir === Val(FFTW.BACKWARD)
        # apply recursively all transforms but last (BFFT!)
        _apply_plans_in_place!(dir, full_plan, nothing, pp...)
        # transpose before last transform
        t = if pp ==()
                nothing
            else
                @assert Base.mightalias(A.arrays[3], A.arrays[2]) # they're aliased!
                t = Transpositions.Transposition(A.arrays[2], A.arrays[3],
                                         method=full_plan.transpose_method)
                transpose!(t, waitall=false)
            end
        # apply separately last transform (BRFFT!)
        _apply_rfft_plan_in_place!(dir, full_plan, A.arrays[1], first(full_plan.plans), A.arrays[2])

        _wait_mpi_operations!(t, full_plan.timer)
    end

    A
end

function _apply_plans_out_of_place!(
        dir::Val, full_plan::PencilFFTPlan, y::PencilArray, x::PencilArray,
        plan::PencilPlan1D, next_plans::Vararg{PencilPlan1D})
    @assert !is_inplace(full_plan) && !is_inplace(plan)
    r = transform_info(dir, plan)

    # Transpose data if required.
    u, t = if pencil(x) === r.Pi
        x, nothing
    else
        u = _temporary_pencil_array(r.Ti, r.Pi, full_plan.ibuf,
                                    full_plan.extra_dims)
        t = Transpositions.Transposition(u, x, method=full_plan.transpose_method)
        u, transpose!(t, waitall=false)
    end

    v = if pencil(y) === r.Po
        y
    else
        _temporary_pencil_array(r.To, r.Po, full_plan.obuf,
                                full_plan.extra_dims)
    end

    @timeit_debug full_plan.timer "FFT" mul!(parent(v), r.fftw_plan, parent(u))

    _wait_mpi_operations!(t, full_plan.timer)
    _apply_plans_out_of_place!(dir, full_plan, y, v, next_plans...)
end

_apply_plans_out_of_place!(dir::Val, ::PencilFFTPlan, y::PencilArray,
                           x::PencilArray) = y

# Wait for send operations to complete (only has an effect for specific
# transposition methods).
_wait_mpi_operations!(t, to) = @timeit_debug to "MPI.Waitall" MPI.Waitall(t)
_wait_mpi_operations!(::Nothing, to) = nothing

function _apply_plans_in_place!(
        dir::Val, full_plan::PencilFFTPlan, u_prev::Union{Nothing, PencilArray},
        pair::PlanArrayPair, next_pairs...)
    plan = pair.first
    u = pair.second
    r = transform_info(dir, plan)

    @assert is_inplace(full_plan) && is_inplace(plan)
    @assert pencil(u) === r.Pi === r.Po

    # Buffers should take no memory for in-place transforms.
    @assert length(full_plan.ibuf) == length(full_plan.obuf) == 0

    t = if u_prev === nothing
        nothing
    else
        # Transpose data from previous configuration.
        @assert Base.mightalias(u_prev, u)  # they're aliased!
        t = Transpositions.Transposition(u, u_prev,
                                         method=full_plan.transpose_method)
        transpose!(t, waitall=false)
    end

    # Perform in-place FFT
    @timeit_debug full_plan.timer "FFT!" r.fftw_plan * parent(u)

    _wait_mpi_operations!(t, full_plan.timer)
    _apply_plans_in_place!(dir, full_plan, u, next_pairs...)
end

_apply_plans_in_place!(::Val, ::PencilFFTPlan, u_prev::PencilArray) = u_prev

function _apply_rfft_plan_in_place!(dir::Val, full_plan::PencilFFTPlan, A_out ::PencilArray{To,N}, p::PencilPlan1D{ti,to,Pi,Po,Tr}, A_in ::PencilArray{Ti,N}) where
    {Ti<:RealOrComplex{T},To<:RealOrComplex{T},ti<:RealOrComplex{T},to<:RealOrComplex{T},Pi,Po,N,Tr<:Union{Transforms.RFFT!,Transforms.BRFFT!}} where T<:FFTReal
   fft_plan = dir === Val(FFTW.FORWARD) ? p.fft_plan : p.bfft_plan
   @timeit_debug full_plan.timer "FFT!" mul!(parent(A_out), fft_plan, parent(A_in)) 
end

_split_first(a, b...) = (a, b)  # (x, y, z, w) -> (x, (y, z, w))

function _make_pairs(plans::Tuple{Vararg{PencilPlan1D,N}},
                     arrays::Tuple{Vararg{PencilArray,N}}) where {N}
    p, p_next = _split_first(plans...)
    a, a_next = _split_first(arrays...)
    (p => a, _make_pairs(p_next, a_next)...)
end

_make_pairs(::Tuple{}, ::Tuple{}) = ()

@inline function _temporary_pencil_array(
        ::Type{T}, p::Pencil, buf::DenseVector{UInt8}, extra_dims::Dims,
    ) where {T}
    # Create "unsafe" pencil array wrapping buffer data.
    dims = (size_local(p, MemoryOrder())..., extra_dims...)
    nb = prod(dims) * sizeof(T)
    resize!(buf, nb)
    x = Transpositions.unsafe_as_array(T, buf, dims)
    PencilArray(p, x)
end

_temporary_pencil_array(::Type, ::Nothing, etc...) = nothing
