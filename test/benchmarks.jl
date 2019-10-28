#!/usr/bin/env julia

using PencilFFTs.Pencils

import Base: show

using MPI
using Profile
using Random
using Test

const PROFILE = true
const PROFILE_OUTPUT = "profile.txt"

const DIMS = (128, 192, 64)
const ITERATIONS = 100

# Benchmark time accumulator.
mutable struct BenchTimes
    iterations  :: Int
    elapsedtime :: UInt64  # elapsed times in ns
    gc_alloc_bytes :: Int
    gc_total_time  :: Int
    gc_alloc_count :: Int
    BenchTimes() = new(0, 0, 0, 0, 0)
end

function show(io::IO, times::BenchTimes)
    # This is the same function called by the @time macro.
    Base.time_print(times.elapsedtime / times.iterations,
                    times.gc_alloc_bytes / times.iterations,
                    times.gc_total_time / times.iterations,
                    times.gc_alloc_count / times.iterations)
    print(io, " [$(times.iterations) iterations]")
end

# Based on the @time implementation.
# Elapsed time is accumulated in the `times` struct.
macro mpi_time(times, ex)
    quote
        $(esc(times)).iterations += 1
        local stats = Base.gc_num()
        local elapsedtime = time_ns()
        local val = $(esc(ex))
        $(esc(times)).elapsedtime += time_ns() - elapsedtime
        local diff = Base.GC_Diff(Base.gc_num(), stats)
        $(esc(times)).gc_alloc_bytes += diff.allocd
        $(esc(times)).gc_total_time += diff.total_time
        $(esc(times)).gc_alloc_count += Base.gc_alloc_count(diff)
        val
    end
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

    times = ntuple(n -> BenchTimes(), 4)

    # Precompile functions
    transpose!(u[2], u[1])
    transpose!(u[3], u[2])
    transpose!(u[2], u[3])
    transpose!(u[1], u[2])

    for it = 1:ITERATIONS
        @mpi_time times[1] transpose!(u[2], u[1])
        @mpi_time times[2] transpose!(u[3], u[2])
        @mpi_time times[3] transpose!(u[2], u[3])
        @mpi_time times[4] transpose!(u[1], u[2])
    end

    @test u[1] == u_orig

    if myrank == 0
        println(
            """
            Processes:          $proc_dims
            Data dimensions:    $data_dims
            Transpositions:""")
        println.(times)
        println()
    end

    if PROFILE
        Profile.clear()
        @profile for it = 1:ITERATIONS
            transpose!(u[2], u[1])
            transpose!(u[3], u[2])
            transpose!(u[2], u[3])
            transpose!(u[1], u[2])
        end
        if myrank == 0
            open(io -> Profile.print(io, maxdepth=6), PROFILE_OUTPUT, "w")
        end
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
