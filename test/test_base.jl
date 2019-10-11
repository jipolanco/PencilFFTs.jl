#!/usr/bin/env julia

using PencilFFTs

using MPI
using Test

function main()
    MPI.Init()

    Nxyz = (16, 32, 24)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    P1 = Nproc > 1 ? 2 : 1
    P2 = Nproc ÷ P1

    plan = PencilPlan(comm, P1, P2, Nxyz...)

    if MPI.Comm_rank(comm) == 0
        show(plan)
    end

    MPI.Finalize()
end

main()
