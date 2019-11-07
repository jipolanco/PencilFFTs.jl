const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal

using LinearAlgebra

## Forward transforms
function mul!(dst::PencilArray{To,N}, p::PencilFFTPlan{T,N},
              src::PencilArray{Ti,N}) where {T, N,
                                             Ti <: RealOrComplex{T},
                                             To <: RealOrComplex{T}}
    @timeit_debug p.timer "PencilFFTs mul!" begin
        _check_arrays(p, src, dst)
        _apply_plans!(dst, src, p.plans...)
    end
end

function *(p::PencilFFTPlan, src::PencilArray)
    @timeit_debug p.timer "PencilFFTs *" begin
        _check_arrays(p, src, nothing)
        dst = allocate_output(p)
        mul!(dst, p, src)
    end
end

function _apply_plans!(y::PencilArray, x::PencilArray, plan::PencilPlan1D,
                       next_plans::Vararg{PencilPlan1D})
    Pi = plan.pencil_in
    Po = plan.pencil_out

    # Transpose pencil if required.
    u = if pencil(x) === Pi
        x
    else
        u = _temporary_pencil_array(Pi, plan.ibuf)
        transpose!(u, x)
    end

    v = pencil(y) === Po ? y : _temporary_pencil_array(Po, plan.obuf)
    @timeit_debug plan.timer "FFT" mul!(data(v), plan.fft_plan, data(u))

    _apply_plans!(y, v, next_plans...)
end

_apply_plans!(y::PencilArray, x::PencilArray) = y

function _check_arrays(p::PencilFFTPlan, xin, xout)
    if xin !== nothing && first(p.plans).pencil_in !== pencil(xin)
        throw(ArgumentError("unexpected dimensions of input data"))
    end
    if xout !== nothing && last(p.plans).pencil_out !== pencil(xout)
        throw(ArgumentError("unexpected dimensions of output data"))
    end
    nothing
end

function _temporary_pencil_array(p::Pencil, buf::Vector{UInt8})
    @timeit_debug p.timer "temporary_pencil_array" begin
        # Create "unsafe" pencil array wrapping buffer data.
        T = eltype(p)
        dims = size_local(p)
        nb = prod(dims) * sizeof(T)
        resize!(buf, nb)
        x = Pencils.unsafe_as_array(T, buf, dims)
        PencilArray(p, x)
    end
end
