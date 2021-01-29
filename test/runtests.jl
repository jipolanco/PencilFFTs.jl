#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using MPI: MPI, mpiexec
using InteractiveUtils: versioninfo

# Load test packages to trigger precompilation
using PencilFFTs

test_files = [
    "taylor_green.jl",
    "rfft.jl",
    "transforms.jl",
]

# Make sure that example files run.
example_dir = joinpath("..", "examples")
example_files = joinpath.(
    example_dir,
    filter(fname -> splitext(fname)[2] == ".jl", readdir(example_dir))
)

Nproc = let N = get(ENV, "JULIA_MPI_TEST_NPROCS", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end
files = [test_files..., example_files...]

println()
versioninfo()
println("\n", MPI.MPI_LIBRARY_VERSION_STRING, "\n")

for fname in files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        # Disable precompilation to prevent race conditions when loading
        # packages.
        run(`$cmd -n $Nproc $(Base.julia_cmd()) $fname`)
    end
    println()
end
