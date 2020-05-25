#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using MPI: mpiexec
using HDF5
using PencilFFTs
using Test

test_files = [
    "hdf5.jl",
    "taylor_green.jl",
    "rfft.jl",
    "pencils.jl",
    "transforms.jl",
]

# Make sure that example files run.
example_dir = joinpath("..", "examples")
example_files = joinpath.(
    example_dir,
    filter(fname -> splitext(fname)[2] == ".jl", readdir(example_dir))
)

Nproc = clamp(Sys.CPU_THREADS, 4, 6)
files = [test_files..., example_files...]

for fname in files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        run(`$cmd -n $Nproc $(Base.julia_cmd()) $fname`)
    end
    println()
end
