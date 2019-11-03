const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal

function mul!(out::PencilArray{To,N}, p::PencilFFTPlan{T,N},
              in::PencilArray{Ti,N}) where {T, N,
                                            Ti <: RealOrComplex{T},
                                            To <: RealOrComplex{T}}
    # TODO remove lots of allocations everywhere!!
    plans = p.plans
    @assert first(plans).pencil_in === pencil(in)
    @assert last(plans).pencil_out === pencil(out)
    u = _apply_plans(in, plans...)
    copy!(out, u)
end

function _apply_plans(u::PencilArray, plan::PencilPlan1D,
                      next_plans::Vararg{PencilPlan1D})
    # We assume that the array already comes transposed for this transform.
    # @assert pencil(u) === plan.pencil_in
    Pi = plan.pencil_in
    Po = plan.pencil_out

    # Transpose pencil if required.
    v = if pencil(u) === Pi
        u
    else
        v = PencilArray(Pi)
        transpose!(v, u)
    end

    w = PencilArray(Po)
    mul!(data(w), plan.fft_plan, data(v))
    _apply_plans(w, next_plans...)
end

_apply_plans(u::PencilArray) = u

# TODO
# function *(p::PencilFFTPlan, in::PencilArray)
# end
