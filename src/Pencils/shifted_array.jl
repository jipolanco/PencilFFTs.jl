# See https://docs.julialang.org/en/latest/devdocs/offset-arrays

"""
    ShiftedArrayView{T,N}

Wraps an array shifting its indices.
"""
struct ShiftedArrayView{T, N,
                        A <: AbstractArray{T, N}} <: AbstractArray{T, N}
    x :: A
    function ShiftedArrayView(x::A) where {T, N, A <: AbstractArray{T, N}}
        new{T, N, A}(x)
    end
end

# TODO...

"""
    global_view(x::PencilArray)

Create a [`ShiftedArrayView`](@ref) of a `PencilArray` that takes global
indices.

The order of indices in the returned view is the same as for the original array
`x`. That is, if the indices of `x` are permuted, so are those of the returned
array.
"""
global_view(x::PencilArray) = ShiftedArrayView(x)
