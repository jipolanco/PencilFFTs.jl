using .HDF5

const HDF5FileOrGroup = Union{HDF5.HDF5File, HDF5.HDF5Group}

"""
    setindex!(
        g::Union{HDF5File,HDF5Group}, x::PencilArray, name::String;
        chunks=false, collective=true,
    )

Write [`PencilArray`](@ref) to parallel HDF5 file.

For performance reasons, the memory layout of the data is conserved. In other
words, if the dimensions of the `PencilArray` are permuted in memory, then the
data is written in permuted form.

# Optional arguments

- if `chunks=true`, data is written in chunks, with roughly one chunk
  per MPI process. This may (or may not) improve performance in parallel
  filesystems.

- if `collective=true`, the dataset is written collectivelly. This is
  usually recommended for performance.

# Example

Open a parallel HDF5 file and write `PencilArray` to the file:

```julia
u = PencilArray(...)

comm = get_comm(u)
info = MPI.Info()

h5open("filename.h5", "w", "fapl_mpio", (comm, info)) do ff
    ff["u", chunks=true] = u
end
```

"""
function Base.setindex!(g::HDF5FileOrGroup, x::PencilArray, name::String;
                        chunks=true, collective=true)
    dims_global = size_global(x, permute=true)

    toclose = true  # property lists should be closed by the GC

    lcpl = HDF5._link_properties(name)  # this is the default in HDF5.jl
    dcpl = p_create(HDF5.H5P_DATASET_CREATE, toclose)
    dapl = HDF5.DEFAULT_PROPERTIES
    dxpl = p_create(HDF5.H5P_DATASET_XFER, toclose)

    if chunks
        chunk = h5_chunk_size(x, permute=true)
        HDF5.set_chunk(dcpl, chunk...)
    end

    if collective
        HDF5.h5p_set_dxpl_mpio(dxpl, HDF5.H5FD_MPIO_COLLECTIVE)
    end

    dset = d_create(
        g, name, datatype(eltype(x)), dataspace(dims_global),
        lcpl, dcpl, dapl, dxpl,
    )

    inds = range_local(x, permute=true)
    dset[inds...] = parent(x)  # this only makes sense if permute=true above

    x
end

function h5_chunk_size(x::PencilArray; permute=true)
    # Determine chunk size for writing to HDF5 dataset.
    # The idea is that each process writes to a single separate chunk of the
    # dataset, of size `dims_local`.
    # This only works if the data is ideally balanced among processes, i.e. if
    # the local dimensions of the dataset are the same for all processes.
    dims_local = size_local(x, permute=permute)

    # In the general case that the data is not well balanced, we take the
    # minimum size along each dimension.
    chunk = MPI.Allreduce(collect(dims_local), min, get_comm(x))

    N = ndims(x)
    @assert length(chunk) == N
    ntuple(d -> chunk[d], Val(N))
end
