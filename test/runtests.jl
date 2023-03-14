#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using MPI: MPI, mpiexec
using InteractiveUtils: versioninfo

# Load test packages to trigger precompilation
using PencilFFTs

test_files = [
    "taylor_green.jl",
    "brfft.jl",
    "rfft.jl",
    "transforms.jl",
]

# Also run some (but not all!) examples.
example_dir = joinpath(@__DIR__, "..", "examples")
example_files = joinpath.(
    example_dir,
    ["gradient.jl", "in-place.jl"]
)

Nproc = let N = get(ENV, "JULIA_MPI_TEST_NPROCS", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end
files = vcat(example_files, test_files)

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
