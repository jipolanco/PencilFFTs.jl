using .HDF5
import Libdl

export ph5open

const HDF5FileOrGroup = Union{HDF5.HDF5File, HDF5.HDF5Group}

const H5MPIHandle = let csize = sizeof(MPI.MPI_Comm)
    @assert csize in (4, 8)
    csize === 4 ? HDF5.Hmpih32 : HDF5.Hmpih64
end

mpi_to_h5_handle(obj::Union{MPI.Comm, MPI.Info}) =
    reinterpret(H5MPIHandle, obj.val)

function h5_to_mpi_handle(
        ::Type{T}, h::H5MPIHandle) where {T <: Union{MPI.Comm, MPI.Info}}
    T(reinterpret(MPI.MPI_Comm, h))
end

const _HAS_PARALLEL_HDF5 = Libdl.dlopen(HDF5.libhdf5) do lib
    Libdl.dlsym(lib, :H5Pget_fapl_mpio, throw_error=false) !== nothing
end

"""
    hdf5_has_parallel() -> Bool

Returns `true` if the loaded HDF5 libraries support MPI I/O.
"""
hdf5_has_parallel() = _HAS_PARALLEL_HDF5

function check_hdf5_parallel()
    hdf5_has_parallel() && return
    error(
        "HDF5.jl has no parallel support." *
        " Make sure that you're using MPI-enabled HDF5 libraries, and that" *
        " MPI was loaded before HDF5." *
        " See https://jipolanco.github.io/PencilFFTs.jl/latest/PencilIO/#Setting-up-Parallel-HDF5-1" *
        " for details."
    )
end

"""
    ph5open([f::Function], filename, [mode="r"], comm::MPI.Comm,
            [info::MPI.Info=MPI.Info()], prop_lists...) -> HDF5.File

Open parallel HDF5 file.

This function is a thin wrapper over `HDF5.h5open`.
It converts MPI.jl types (`MPI.Comm` and `MPI.Info`) to their counterparts in
HDF5.jl.
It also throws an informative error if the loaded HDF5 libraries do not include
parallel support.

# Property lists

This function automatically sets the
[`fapl_mpio`](https://portal.hdfgroup.org/display/HDF5/H5P_SET_FAPL_MPIO) file
access property list to the given MPI communicator and info object.
Other property lists should be given as name-value pairs, following the
[`h5open`
syntax](https://github.com/JuliaIO/HDF5.jl/blob/master/doc/hdf5.md#passing-parameters).

Property lists are passed to
[`h5f_create`](https://portal.hdfgroup.org/display/HDF5/H5F_CREATE).
The following property types are recognised:
- [file creation properties](https://portal.hdfgroup.org/display/HDF5/File+Creation+Properties),
- [file access properties](https://portal.hdfgroup.org/display/HDF5/File+Access+Properties).
"""
function ph5open(
        filename::AbstractString, mode::AbstractString,
        comm::MPI.Comm, info::MPI.Info = MPI.Info(),
        plist_pairs...,
    )
    check_hdf5_parallel()
    fapl_mpio = mpi_to_h5_handle.((comm, info))
    h5open(filename, mode, "fapl_mpio", fapl_mpio, plist_pairs...)
end

ph5open(filename::AbstractString, comm::MPI.Comm, args...; kwargs...) =
    ph5open(filename, "r", comm, args...; kwargs...)

function ph5open(f::Function, args...; kwargs...)
    fid = ph5open(args...; kwargs...)
    try
        f(fid)
    finally
        close(fid)
    end
end

"""
    setindex!(g::Union{HDF5File,HDF5Group}, x::MaybePencilArrayCollection,
              name::String, prop_lists...; chunks=false, collective=true)

Write [`PencilArray`](@ref) or [`PencilArrayCollection`](@ref) to parallel HDF5
file.

For performance reasons, the memory layout of the data is conserved. In other
words, if the dimensions of a `PencilArray` are permuted in memory, then the
data is written in permuted form.

In the case of a `PencilArrayCollection`, each array of the collection is written
as a single component of a higher-dimension dataset.

# Optional arguments

- if `chunks=true`, data is written in chunks, with roughly one chunk
  per MPI process. This may (or may not) improve performance in parallel
  filesystems.

- if `collective=true`, the dataset is written collectivelly. This is
  usually recommended for performance.

- additional property lists may be specified by name-value pairs in
  `prop_lists`, following the [HDF5.jl
  syntax](https://github.com/JuliaIO/HDF5.jl/blob/master/doc/hdf5.md#passing-parameters).
  These property lists take precedence over keyword arguments.
  For instance, if the `"dxpl_mpio", HDF5.H5FD_MPIO_COLLECTIVE` pair is passed,
  then the value of the `collective` argument is ignored.

# Property lists

Property lists are passed to
[`h5d_create`](https://portal.hdfgroup.org/display/HDF5/H5D_CREATE2)
and [`h5d_write`](https://portal.hdfgroup.org/display/HDF5/H5D_WRITE).
The following property types are recognised:
- [link creation properties](https://portal.hdfgroup.org/display/HDF5/Attribute+and+Link+Creation+Properties),
- [dataset creation properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Creation+Properties),
- [dataset access properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Access+Properties),
- [dataset transfer properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Transfer+Properties).

# Example

Open a parallel HDF5 file and write some `PencilArray`s to the file:

```julia
u = PencilArray(...)
v = PencilArray(...)

comm = get_comm(u)
info = MPI.Info()

ph5open("filename.h5", "w", comm, info) do ff
    ff["u", chunks=true] = u
    ff["uv"] = (u, v)  # this is a two-component PencilArrayCollection (assuming equal dimensions of `u` and `v`)
end
```

"""
function Base.setindex!(
        g::HDF5FileOrGroup, x::MaybePencilArrayCollection,
        name::String, prop_pairs...;
        chunks=false, collective=true,
    )
    check_phdf5_file(g, x)

    # Add extra property lists if required by keyword args.
    props = collect(Any, prop_pairs)

    if chunks && "chunk" ∉ prop_pairs
        chunk = h5_chunk_size(x, permute=true)
        push!(props, "chunk", chunk)
    end

    if collective && "dxpl_mpio" ∉ prop_pairs
        push!(props, "dxpl_mpio", HDF5.H5FD_MPIO_COLLECTIVE)
    end

    dims_global = h5_dataspace_dims(x)
    dset = d_create(g, name, h5_datatype(x), dataspace(dims_global), props...)
    inds = range_local(x, permute=true)
    to_hdf5(dset, x, inds)

    x
end

"""
    read!(g::Union{HDF5File,HDF5Group}, x::MaybePencilArrayCollection,
          name::String, prop_lists...; collective=true)

Read [`PencilArray`](@ref) or [`PencilArrayCollection`](@ref) from parallel HDF5
file.

See [`setindex!`](@ref) for details on optional arguments.

# Property lists

Property lists are passed to
[`h5d_open`](https://portal.hdfgroup.org/display/HDF5/H5D_OPEN2)
and [`h5d_read`](https://portal.hdfgroup.org/display/HDF5/H5D_READ).
The following property types are recognised:
- [dataset access properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Access+Properties),
- [dataset transfer properties](https://portal.hdfgroup.org/display/HDF5/Dataset+Transfer+Properties).

# Example

Open a parallel HDF5 file and read some `PencilArray`s:

```julia
u = PencilArray(...)
v = PencilArray(...)

comm = get_comm(u)
info = MPI.Info()

ph5open("filename.h5", "r", comm, info) do ff
    read!(ff, u, "u")
    read!(ff, (u, v), "uv")
end
```
"""
function Base.read!(g::HDF5FileOrGroup, x::MaybePencilArrayCollection,
                    name::AbstractString, prop_pairs...; collective=true)
    dapl = p_create(HDF5.H5P_DATASET_ACCESS, false, prop_pairs...)
    dxpl = p_create(HDF5.H5P_DATASET_XFER, false, prop_pairs...)

    # Add extra property lists if required by keyword args.
    if collective && "dxpl_mpio" ∉ prop_pairs
        HDF5.h5p_set_dxpl_mpio(dxpl.id, HDF5.H5FD_MPIO_COLLECTIVE)
    end

    dims_global = h5_dataspace_dims(x)
    dset = d_open(g, name, dapl, dxpl)
    check_phdf5_file(parent(dset), x)

    if dims_global != size(dset)
        throw(DimensionMismatch(
            "incompatible dimensions of HDF5 dataset and PencilArray"))
    end

    inds = range_local(x, permute=true)
    from_hdf5!(dset, x, inds)

    x
end

function check_phdf5_file(g, x)
    check_hdf5_parallel()

    plist_id = HDF5.h5f_get_access_plist(file(g))
    plist = HDF5Properties(plist_id, true, HDF5.H5P_FILE_ACCESS)

    # Get HDF5 ids of MPIO driver and of the actual driver being used.
    driver_mpio = ccall((:H5FD_mpio_init, HDF5.libhdf5), HDF5.Hid, ())
    driver = HDF5.h5p_get_driver(plist)
    if driver !== driver_mpio
        error("HDF5 file was not opened with the MPIO driver")
    end

    comm, info = _h5p_get_fapl_mpio(plist)
    if isdefined(MPI, :Comm_compare)  # requires recent MPI.jl
        if MPI.Comm_compare(comm, get_comm(x)) ∉ (MPI.IDENT, MPI.CONGRUENT)
            throw(ArgumentError(
                "incompatible MPI communicators of HDF5 file and PencilArray"
            ))
        end
    end

    close(plist)
    nothing
end

to_hdf5(dset, x::PencilArray, inds) = dset[inds...] = parent(x)

function from_hdf5!(dset, x::PencilArray, inds)
    u = parent(x)

    if stride(u, 1) != 1
        u .= dset[inds...]  # short and easy version (but allocates!)
        return x
    end

    # The following is adapted from one of the _getindex() in HDF5.jl.
    HDF5Scalar = HDF5.HDF5Scalar
    T = eltype(x)
    if !(T <: Union{HDF5Scalar, Complex{<:HDF5Scalar}})
        error("Dataset indexing (hyperslab) is available only for bits types")
    end

    dsel_id = HDF5.hyperslab(dset, inds...)
    memtype = HDF5.datatype(u)
    memspace = HDF5.dataspace(u)

    try
        # This only works for stride-1 arrays.
        HDF5.h5d_read(dset.id, memtype.id, memspace.id, dsel_id, dset.xfer, u)
    finally
        close(memtype)
        close(memspace)
        HDF5.h5s_close(dsel_id)
    end

    x
end

# Define variants for collections.
for func in (:from_hdf5!, :to_hdf5)
    @eval function $func(dset, col::PencilArrayCollection, inds_in)
        for I in CartesianIndices(collection_size(col))
            inds = (inds_in..., Tuple(I)...)
            $func(dset, col[I], inds)
        end
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

function _h5p_get_fapl_mpio(fapl_id)
    h5comm, h5info = HDF5.h5p_get_fapl_mpio(fapl_id, H5MPIHandle)
    comm = h5_to_mpi_handle(MPI.Comm, h5comm)
    info = h5_to_mpi_handle(MPI.Info, h5info)
    comm, info
end
