# TODO
# - create array wrapper including pencil info

struct PencilArray{T, P<:Pencil}
    pencil :: P
    data   :: Array{T,3}
end

"""
    allocate(p::Pencil, [T=Float64])

Allocate uninitialised 3D array with the dimensions of the given pencil.

Data dimensions are permuted if the pencil was defined with a given permutation.
"""
allocate(p::Pencil, ::Type{T}=Float64) where T = Array{T}(undef, size_local(p))

"""
    gather(p::Pencil, x::AbstractArray, root::Integer)

Gather data from all MPI processes into one (big) array.

Data is received by the `root` process.

This can be useful for testing, but it shouldn't be used with very large
datasets!
"""
function gather(p::Pencil, x::AbstractArray, root::Integer)
    rank = MPI.Comm_rank(p.topology.comm)
    # TODO!!
    nothing
end
