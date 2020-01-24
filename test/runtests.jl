#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using FFTW  # this avoids issues with precompilation of FFTW in parallel...
import MPI: mpiexec

const TEST_FILES = [
    "taylor_green.jl",
    "rfft.jl",
    "pencils.jl",
    "transforms.jl",
]

# Make sure that example files run.
const EXAMPLE_DIR = joinpath("..", "examples")
const EXAMPLE_FILES = joinpath.(
    EXAMPLE_DIR,
    filter(fname -> splitext(fname)[2] == ".jl", readdir(EXAMPLE_DIR))
)

function main()
    Nproc = clamp(Sys.CPU_THREADS, 4, 8)
    julia_exec = joinpath(Sys.BINDIR, Base.julia_exename())

    files = [TEST_FILES..., EXAMPLE_FILES...]

    for fname in files
        @info "Running $fname with $Nproc processes..."
        run(`$mpiexec -n $Nproc $julia_exec $fname`)
        println()
    end

    nothing
end

main()
