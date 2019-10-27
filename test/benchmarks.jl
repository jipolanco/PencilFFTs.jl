#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI
using Random
using Test

const DIMS = (128, 192, 64)
const ITERATIONS = 100

struct BenchTimes
    iterations :: Ref{Int}
    transpositions :: Matrix{Float64}
    BenchTimes() = new(0, zeros(2, 2))
end

function benchmark_decomp(comm, proc_dims::Tuple, data_dims::Tuple)
    topo = Topology(comm, proc_dims)
    M = length(proc_dims)
    @assert M in (1, 2)

    if M == 1  # slab decomposition
        pen1 = Pencil(topo, data_dims, (2, ))
        pen2 = Pencil(pen1, (3, ), permute=(2, 1, 3))
        pen3 = Pencil(pen2, (1, ), permute=(3, 2, 1))
    elseif M == 2  # pencil decomposition
        pen1 = Pencil(topo, data_dims, (2, 3))
        pen2 = Pencil(pen1, (1, 3), permute=(2, 1, 3))
        pen3 = Pencil(pen2, (1, 2), permute=(3, 2, 1))
    end

    u = PencilArray.((pen1, pen2, pen3), Float64)

    myrank = MPI.Comm_rank(comm)
    rng = MersenneTwister(42 + myrank)
    randn!(rng, u[1])
    u[1] .+= 10 * myrank
    u_orig = copy(u[1])

    times = BenchTimes()

    for it = 1:ITERATIONS
        times.iterations[] += 1

        # TODO create macro
        let t0 = MPI.Wtime()
            transpose!(u[2], u[1])
            times.transpositions[1, 1] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u[3], u[2])
            times.transpositions[2, 1] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u[2], u[3])
            times.transpositions[1, 2] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u[1], u[2])
            times.transpositions[2, 2] += MPI.Wtime() - t0
        end
    end

    @test u[1] == u_orig

    times.transpositions ./= times.iterations[]

    if myrank == 0
        @show proc_dims
        @show data_dims
        @show times.iterations[]
        @show times.transpositions
        println()
    end

    nothing
end

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    # Slab decompositions
    benchmark_decomp(comm, (Nproc, ), DIMS)

    # Pencil decompositions
    benchmark_decomp(comm, (1, Nproc), DIMS)
    benchmark_decomp(comm, (Nproc, 1), DIMS)

    # Let MPI_Dims_create choose the decomposition.
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    benchmark_decomp(comm, proc_dims, DIMS)

    MPI.Finalize()
end

main()
