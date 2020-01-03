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

function Base.LinearIndices(g::GlobalPencilArray)
    A = parent(g)
    off = g.offsets
    PermutedLinearIndices(LinearIndices(parent(A)), get_permutation(A), off)
end
