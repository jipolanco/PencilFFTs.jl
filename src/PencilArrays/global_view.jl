"""
    GlobalPencilArray{T,N} <: AbstractArray{T,N}

Alias for an `OffsetArray` wrapping a [`PencilArray`](@ref).

Unlike `PencilArray`s, `GlobalPencilArray`s take *global* indices, which
in general don't start at 1 for a given MPI process.

The [`global_view`](@ref) function should be used to create a
`GlobalPencilArray` from a `PencilArray`.
"""
const GlobalPencilArray{T,N} = OffsetArray{T,N,A} where {A <: PencilArray}

"""
    global_view(x::PencilArray)

Create an [`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) of a
`PencilArray` that takes unpermuted global indices.
"""
function global_view(x::PencilArray)
    r = range_local(x, permute=false)
    offsets = first.(r) .- 1
    xo = OffsetArray(x, offsets)
    @assert parent(xo) === x  # OffsetArray shouldn't create a copy...
    xo :: GlobalPencilArray
end

# Account for index permutation in global views.
@inline Base._sub2ind(x::GlobalPencilArray, I...) =
    Base._sub2ind(parent(x), (I .- x.offsets)...)
