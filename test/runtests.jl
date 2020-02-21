#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

import MPI: mpiexec_path

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

# Code coverage command line options; must correspond to src/julia.h
# and src/ui/repl.c
# (Copied from MPI.jl)
const JL_LOG_NONE = 0
const JL_LOG_USER = 1
const JL_LOG_ALL = 2
const COVERAGE_OPTS =
    Dict{Int, String}(JL_LOG_NONE => "none",
                      JL_LOG_USER => "user",
                      JL_LOG_ALL => "all")

function main()
    Nproc = clamp(Sys.CPU_THREADS, 4, 8)
    julia_exec = joinpath(Sys.BINDIR, Base.julia_exename())

    files = [TEST_FILES..., EXAMPLE_FILES...]
    cov = COVERAGE_OPTS[Base.JLOptions().code_coverage]
    julia_args = ["--compiled-modules=no", "--code-coverage=$cov"]

    for fname in files
        @info "Running $fname with $Nproc processes..."
        run(`$mpiexec_path -n $Nproc $julia_exec $julia_args $fname`)
        println()
    end

    nothing
end

main()
