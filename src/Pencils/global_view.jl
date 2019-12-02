const GlobalPencilArray{T,N} = OffsetArray{T,N,A} where {A <: PencilArray}

# Base.IndexStyle(::Type{<:GlobalPencilArray}) = IndexCartesian()

"""
    global_view(x::PencilArray)

Create an [`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) of a
`PencilArray` that takes global permuted indices.
"""
function global_view(x::PencilArray)
    r = range_local(x, permute=true)
    offsets = first.(r) .- 1
    xo = OffsetArray(x, offsets)
    @assert parent(xo) === x  # OffsetArray shouldn't create a copy...
    xo :: GlobalPencilArray
end

function spatial_indices(x::GlobalPencilArray)
    Np = ndims(x) - ndims_extra(parent(x))
    CartesianIndices(ntuple(n -> axes(x, n), Val(Np)))
end
