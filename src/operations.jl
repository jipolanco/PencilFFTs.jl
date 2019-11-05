const RealOrComplex{T} = Union{T, Complex{T}} where T <: FFTReal

using LinearAlgebra

function mul!(out::PencilArray{To,N}, p::PencilFFTPlan{T,N},
              in::PencilArray{Ti,N}) where {T, N,
                                            Ti <: RealOrComplex{T},
                                            To <: RealOrComplex{T}}
    # TODO remove lots of allocations everywhere!!
    _check_arrays(p, in, out)
    _apply_plans!(out, in, p.plans...)
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
        u = _temporary_pencil_array(Pi, plan.ibuf)
        transpose!(u, x)
    end

    v = pencil(y) === Po ? y : _temporary_pencil_array(Po, plan.obuf)
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

function _temporary_pencil_array(p::Pencil, buf::Vector{UInt8})
    # Create "unsafe" pencil array wrapping buffer data.
    T = eltype(p)
    dims = size_local(p)
    nb = prod(dims) * sizeof(T)
    resize!(buf, nb)
    x = Pencils.unsafe_as_array(T, buf, dims)
    PencilArray(p, x)
end
