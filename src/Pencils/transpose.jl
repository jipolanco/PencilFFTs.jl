"""
    transpose!(out::AbstractArray{T}, Pout::Pencil{S}, Pin::Pencil{D},
               in::AbstractArray{T}) where {T, D, S}

Transpose data from the `Pin` to the `Pout` pencil configuration.
"""
function transpose!(out::AbstractArray{T}, Pout::Pencil{S},
                    in::AbstractArray{T}, Pin::Pencil{D}) where {T, D, S}
    assert_compatible(Pin, in)
    assert_compatible(Pout, out)
    transpose_impl!(out, Pout, in, Pin)
end

function assert_compatible(p::Pencil, x::AbstractArray)
    if ndims(x) != 3
        # TODO allow ndims > 3
        throw(ArgumentError("Array must have 3 dimensions."))
    end
    dims = size_local(p)
    if size(x)[1:3] !== dims
        throw(ArgumentError("Array must have dimensions $dims"))
    end
    nothing
end

function assert_compatible(p::Pencil, q::Pencil)
    if p.topology !== q.topology
        throw(ArgumentError("Pencil topologies must be the same."))
    end
    if p.size_global !== q.size_global
        throw(ArgumentError("Global data sizes must be the same between " *
                            "different pencil configurations."))
    end
    nothing
end

# Transpose Pencil{1} -> Pencil{2}
function transpose_impl!(out::AbstractArray{T}, Pout::Pencil{2},
                         in::AbstractArray{T}, Pin::Pencil{1}) where T
    out
end
