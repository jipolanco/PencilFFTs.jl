using .HDF5

const HDF5FileOrGroup = Union{HDF5.HDF5File, HDF5.HDF5Group}

"""
    setindex!(
        g::Union{HDF5File,HDF5Group}, x::MaybePencilArrayCollection, name::String;
        chunks=false, collective=true,
    )

Write [`PencilArray`](@ref) or [`PencilArrayCollection`](@ref) to parallel HDF5
file.

For performance reasons, the memory layout of the data is conserved. In other
words, if the dimensions of a `PencilArray` are permuted in memory, then the
data is written in permuted form.

In the case of a `PencilArrayCollection`, each array of the collection is written
as a single component of a higher dimension dataset.

# Optional arguments

- if `chunks=true`, data is written in chunks, with roughly one chunk
  per MPI process. This may (or may not) improve performance in parallel
  filesystems.

- if `collective=true`, the dataset is written collectivelly. This is
  usually recommended for performance.

# Example

Open a parallel HDF5 file and write some `PencilArray`s to the file:

```julia
u = PencilArray(...)
v = PencilArray(...)

comm = get_comm(u)
info = MPI.Info()

h5open("filename.h5", "w", "fapl_mpio", (comm, info)) do ff
    ff["u", chunks=true] = u
    ff["uv"] = (u, v)  # this is a PencilArrayCollection
end
```

"""
function Base.setindex!(
        g::HDF5FileOrGroup, x::MaybePencilArrayCollection, name::String;
        chunks=true, collective=true,
    )
    check_phdf5_file(g, x)
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

    dims_global = h5_dataspace_dims(x)

    dset = d_create(
        g, name, h5_datatype(x), dataspace(dims_global),
        lcpl, dcpl, dapl, dxpl,
    )

    inds = range_local(x, permute=true)
    to_hdf5(dset, x, inds)

    x
end

function check_phdf5_file(g, x)
    if !HDF5.has_parallel()
        error(
            "HDF5.jl has no parallel support." *
            " Make sure that you're using MPI-enabled HDF5 libraries, and that" *
            " MPI was loaded before HDF5."
        )
    end

    plist_id = HDF5.h5f_get_access_plist(file(g))
    plist = HDF5Properties(plist_id, true, HDF5.H5P_FILE_ACCESS)

    # Get HDF5 ids of MPIO driver and of the actual driver being used.
    driver_mpio = ccall((:H5FD_mpio_init, HDF5.libhdf5), HDF5.Hid, ())
    driver = HDF5.h5p_get_driver(plist)
    if driver !== driver_mpio
        error("HDF5 file was not opened with the MPIO driver")
    end

    comm, info = HDF5.h5p_get_fapl_mpio(plist)
    if MPI.Comm_compare(comm, get_comm(x)) ∉ (MPI.IDENT, MPI.CONGRUENT)
        throw(ArgumentError(
            "incompatible MPI communicators of HDF5 file and PencilArray"
        ))
    end

    close(plist)
    nothing
end

# get_driver()

to_hdf5(dset, x::PencilArray, inds) =
    dset[inds...] = parent(x)

function to_hdf5(dset, col::PencilArrayCollection, inds_in)
    for I in CartesianIndices(collection_size(col))
        inds = (inds_in..., Tuple(I)...)
        to_hdf5(dset, col[I], inds)
    end
end

h5_datatype(x::PencilArray) = datatype(eltype(x))
h5_datatype(x::PencilArrayCollection) = h5_datatype(first(x))

h5_dataspace_dims(x::PencilArray) = size_global(x, permute=true)
h5_dataspace_dims(x::PencilArrayCollection) =
    (h5_dataspace_dims(first(x))..., collection_size(x)...)

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

h5_chunk_size(x::PencilArrayCollection; kwargs...) =
    (h5_chunk_size(first(x); kwargs...)..., collection_size(x)...)
