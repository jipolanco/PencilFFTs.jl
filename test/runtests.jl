#!/usr/bin/env julia

# This is based on the runtests.jl file of MPI.jl.

import MPI: mpiexec

const TEST_FILES = ("pencils.jl", "test_base.jl", )

function main()
    Nproc = clamp(Sys.CPU_THREADS, 4, 8)
    julia_exec = joinpath(Sys.BINDIR, Base.julia_exename())

    for fname in TEST_FILES
        run(`$mpiexec -n $Nproc $julia_exec $fname`)
    end

    nothing
end

main()
