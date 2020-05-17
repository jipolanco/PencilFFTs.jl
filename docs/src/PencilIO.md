# [Parallel I/O](@id PencilIO_module)

```@meta
CurrentModule = PencilFFTs.PencilIO
```

The `PencilFFTs.PencilIO` module contains functions for saving and loading
[`PencilArray`](@ref)s to disk.

Parallel I/O is currently performed using [Parallel
HDF5](https://portal.hdfgroup.org/display/HDF5/Parallel+HDF5)
via the [HDF5.jl](https://github.com/JuliaIO/HDF5.jl) package.

## Setting-up Parallel HDF5

Parallel HDF5 is not enabled in the default installation of HDF5.jl.
For Parallel HDF5 to work, the HDF5 C libraries wrapped by HDF5.jl must be
compiled with parallel support and linked to the specific MPI implementation
that will be used for parallel I/O.
HDF5.jl must be explicitly instructed to use parallel-enabled HDF5 libraries
available in the system.
Similarly, MPI.jl must be instructed to use the corresponding MPI libraries.
This is detailed in the sections below.

Parallel-enabled HDF5 libraries are usually included in computing clusters and
linked to the available MPI implementations.
They are also available via the package manager of a number of Linux
distributions.
(For instance, Fedora includes the `hdf5-mpich-devel` and `hdf5-openmpi-devel`
packages, respectively linked to the MPICH and OpenMPI libraries in the Fedora
repositories.)

The following step-by-step guide assumes one already has access to
parallel-enabled HDF5 libraries linked to an existent MPI installation.

### 1. Using system-provided MPI libraries

Set the environment variable `JULIA_MPI_BINARY=system` and then run
`]build MPI` from Julia.
For more control, one can also set the `JULIA_MPI_PATH` environment variable to
the top-level installation directory of the MPI library.

See the [MPI.jl
docs](https://juliaparallel.github.io/MPI.jl/stable/configuration/#Using-a-system-provided-MPI-1)
for details.

### 2. Using parallel HDF5 libraries

Set the `JULIA_HDF5_LIBRARY_PATH` environment variable to the directory where
the HDF5 libraries compiled with parallel support are found.
Then run `]build HDF5` from Julia.
Note that the selected HDF5 library must be linked to the MPI library chosen in
the previous section.
For the set-up to be persistent across HDF5.jl updates, consider setting
`JULIA_HDF5_LIBRARY_PATH` in `~/.bashrc` or similar.

See the [HDF5.jl
README](https://github.com/JuliaIO/HDF5.jl#installation) for details.

### 3. Loading PencilIO

In the `PencilIO` module, the HDF5.jl package is lazy-loaded
using [Requires](https://github.com/JuliaPackaging/Requires.jl).
This means that, in Julia code, `PencilFFTs` must be loaded *after* `HDF5` for
parallel I/O functionality to be available.

The following order of `using`s ensures that parallel I/O support is available:

```julia
using MPI
using HDF5
using PencilFFTs
```

## TODO

- reading files

- example

- MPI hints

## Library

```@docs
ph5open
hdf5_has_parallel
setindex!
read!
```
