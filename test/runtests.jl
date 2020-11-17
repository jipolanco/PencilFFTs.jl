#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using MPI: mpiexec

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

Nproc = let N = get(ENV, "JULIA_MPI_NPROC", nothing)
    N === nothing ? clamp(Sys.CPU_THREADS, 4, 6) : parse(Int, N)
end
files = [test_files..., example_files...]

for fname in files
    @info "Running $fname with $Nproc processes..."
    mpiexec() do cmd
        # Disable precompilation to prevent race conditions when loading
        # packages.
        run(`$cmd -n $Nproc $(Base.julia_cmd()) --compiled-modules=no $fname`)
    end
    println()
end
