module PencilIO

using ..PencilArrays
import ..PencilArrays: MaybePencilArrayCollection, collection_size

using MPI
using Requires: @require

function __init__()
    @require HDF5="f67ccb44-e63f-5c2f-98bd-6dc0ccc4ba2f" @eval include("hdf5.jl")
end

end
