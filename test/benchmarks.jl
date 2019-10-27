#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI
using Random
using Test

const DIMS = (128, 192, 64)
const ITERATIONS = 100

mutable struct BenchTimes
    iterations :: Int
    transpositions :: Vector{Float64}
    BenchTimes() = new(0, zeros(4))
end

function benchmark_decomp(comm, proc_dims::Tuple, data_dims::Tuple)
    topo = Topology(comm, proc_dims)

    pen1 = Pencil{1}(topo, data_dims)
    pen2 = Pencil{2}(pen1, permute=(2, 1, 3))
    pen3 = Pencil{3}(pen1, permute=(3, 2, 1))

    u1 = PencilArray(pen1, Float64)
    u2 = PencilArray(pen2, Float64)
    u3 = PencilArray(pen3, Float64)

    myrank = MPI.Comm_rank(comm)
    rng = MersenneTwister(42 + myrank)
    randn!(rng, u1)
    u1 .+= 10 * myrank

    times = BenchTimes()

    for it = 1:ITERATIONS
        times.iterations += 1

        # TODO create macro
        let t0 = MPI.Wtime()
            transpose!(u2, u1)
            times.transpositions[1] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u3, u2)
            times.transpositions[2] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u2, u3)
            times.transpositions[3] += MPI.Wtime() - t0
        end

        let t0 = MPI.Wtime()
            transpose!(u1, u2)
            times.transpositions[4] += MPI.Wtime() - t0
        end
    end

    times.transpositions ./= times.iterations

    if myrank == 0
        @show proc_dims
        @show data_dims
        @show times.iterations
        @show times.transpositions
        println()
    end

    nothing
end

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    # Let MPI_Dims_create choose the decomposition.
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    benchmark_decomp(comm, proc_dims, DIMS)

    MPI.Finalize()
end

main()
