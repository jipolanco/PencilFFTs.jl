const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal

using LinearAlgebra

function mul!(out::PencilArray{To,N}, p::PencilFFTPlan{T,N},
              in::PencilArray{Ti,N}) where {T, N,
                                            Ti <: RealOrComplex{T},
                                            To <: RealOrComplex{T}}
    # TODO remove lots of allocations everywhere!!
    _check_arrays(p, in, out)
    plans = p.plans
    _apply_plans!(out, in, plans...)
end

function *(p::PencilFFTPlan, in::PencilArray)
    _check_arrays(p, in)
    out = allocate_output(p)
    mul!(out, p, in)
end

function _apply_plans!(y::PencilArray, x::PencilArray, plan::PencilPlan1D,
                       next_plans::Vararg{PencilPlan1D})
    Pi = plan.pencil_in
    Po = plan.pencil_out

    # Transpose pencil if required.
    u = if pencil(x) === Pi
        x
    else
        u = PencilArray(Pi)
        transpose!(u, x)
    end

    v = pencil(y) === Po ? y : PencilArray(Po)
    mul!(data(v), plan.fft_plan, data(u))

    _apply_plans!(y, v, next_plans...)
end

_apply_plans!(y::PencilArray, x::PencilArray) = y

function _check_arrays(p::PencilFFTPlan, in::PencilArray, out=nothing)
    if first(p.plans).pencil_in !== pencil(in)
        throw(ArgumentError("unexpected dimensions of input data"))
    end
    if out !== nothing && last(p.plans).pencil_out !== pencil(out)
        throw(ArgumentError("unexpected dimensions of output data"))
    end
    nothing
end
