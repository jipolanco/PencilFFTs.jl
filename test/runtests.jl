#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

using FFTW  # this avoids issues with precompilation of FFTW in parallel...
import MPI: mpiexec

const TEST_FILES = (
    # "test_base.jl",
    "pencils.jl",
    "benchmarks.jl",
)

function main()
    Nproc = clamp(Sys.CPU_THREADS, 4, 8)
    julia_exec = joinpath(Sys.BINDIR, Base.julia_exename())

    for fname in TEST_FILES
        @info "Running $fname with $Nproc processes..."
        run(`$mpiexec -n $Nproc $julia_exec $fname`)
    end

    nothing
end

main()
