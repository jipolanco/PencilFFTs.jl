const GlobalPencilArray{T,N} = OffsetArray{T,N,A} where {A <: PencilArray}

Base.IndexStyle(::Type{<:GlobalPencilArray}) = IndexCartesian()

"""
    global_view(x::PencilArray)

Create an [`OffsetArray`](https://github.com/JuliaArrays/OffsetArrays.jl) of a
`PencilArray` that takes global unpermuted indices.
"""
function global_view(x::PencilArray)
    r = range_local(x, permute=false)
    offsets = first.(r) .- 1
    xo = OffsetArray(x, offsets)
    @assert parent(xo) === x  # OffsetArray shouldn't create a copy...
    xo :: GlobalPencilArray
end

function spatial_indices(x::GlobalPencilArray{T,N} where T) where {N}
    p = parent(x)
    E = ndims_extra(p)
    Np = N - E
    CartesianIndices(ntuple(n -> axes(x, n), Val(Np)))
end
