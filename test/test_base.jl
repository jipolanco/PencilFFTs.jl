#!/usr/bin/env julia

using PencilFFTs

using MPI
using Test

# TODO
# - test transforms of vector fields

function test_plan(::Type{T}; extra_dims=()) where T
    Nxyz = (16, 32, 24)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    P1 = Nproc > 1 ? 2 : 1
    P2 = Nproc ÷ P1

    plan = PencilPlan(T, comm, P1, P2, Nxyz...)

    u = allocate_input(plan, extra_dims...)
    uF = allocate_output(plan, extra_dims...)

    if MPI.Comm_rank(comm) == 0
        @show plan
        @show summary(u)
        @show summary(uF)
        println()
    end

    let (rx, ry, rz) = input_range(plan)
        @test size(u) == (length.((rx, ry, rz))..., extra_dims...)
    end

    let (rx, ry, rz) = output_range(plan)
        @test size(uF) == (length.((rz, ry, rx))..., extra_dims...)
    end
end

function main()
    MPI.Init()

    for T in (Float32, Float64)
        for extra_dims in ((), (3, ), (3, 4, ))
            test_plan(T, extra_dims=extra_dims)
        end
    end

    MPI.Finalize()
end

main()
