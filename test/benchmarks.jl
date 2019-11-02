#!/usr/bin/env julia

using PencilFFTs.Pencils

import Base: show

using MPI

using InteractiveUtils
using Profile
using Random
using Test

const PROFILE = false
const PROFILE_OUTPUT = "profile.txt"
const PROFILE_DEPTH = 8

const MEASURE_GATHER = false

const DIMS = (128, 192, 64)
const ITERATIONS = 20

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
    # TODO stuff should be printed to `io`
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

# Slab decomposition
function create_pencils(topo::MPITopology{1}, data_dims, permutation::Val{true})
    pen1 = Pencil(topo, data_dims, (2, ))
    pen2 = Pencil(pen1, decomp_dims=(3, ), permute=(2, 1, 3))
    pen3 = Pencil(pen2, decomp_dims=(2, ), permute=(3, 2, 1))
    pen1, pen2, pen3
end

function create_pencils(topo::MPITopology{1}, data_dims, permutation::Val{false})
    pen1 = Pencil(topo, data_dims, (2, ))
    pen2 = Pencil(pen1, decomp_dims=(3, ))
    pen3 = Pencil(pen2, decomp_dims=(2, ))
    pen1, pen2, pen3
end

# Pencil decomposition
function create_pencils(topo::MPITopology{2}, data_dims, permutation::Val{true})
    pen1 = Pencil(topo, data_dims, (2, 3))
    pen2 = Pencil(pen1, decomp_dims=(1, 3), permute=(2, 1, 3))
    pen3 = Pencil(pen2, decomp_dims=(1, 2), permute=(3, 2, 1))
    pen1, pen2, pen3
end

function create_pencils(topo::MPITopology{2}, data_dims, permutation::Val{false})
    pen1 = Pencil(topo, data_dims, (2, 3))
    pen2 = Pencil(pen1, decomp_dims=(1, 3))
    pen3 = Pencil(pen2, decomp_dims=(1, 2))
    pen1, pen2, pen3
end

function benchmark_decomp(comm, proc_dims::Tuple, data_dims::Tuple;
                          iterations=ITERATIONS,
                          with_permutations::Val=Val(true),
                          extra_dims::Tuple=())
    topo = MPITopology(comm, proc_dims)
    M = length(proc_dims)
    @assert M in (1, 2)

    pens = create_pencils(topo, data_dims, with_permutations)

    u = PencilArray.(pens, extra_dims...)

    myrank = MPI.Comm_rank(comm)
    rng = MersenneTwister(42 + myrank)
    randn!(rng, u[1])
    u[1] .+= 10 * myrank
    u_orig = copy(u[1])

    times = ntuple(n -> BenchTimes(), 5)

    # Precompile functions
    transpose!(u[2], u[1])
    transpose!(u[3], u[2])
    transpose!(u[2], u[3])
    transpose!(u[1], u[2])

    for it = 1:iterations
        @mpi_time times[1] transpose!(u[2], u[1])
        @mpi_time times[2] transpose!(u[3], u[2])
        @mpi_time times[3] transpose!(u[2], u[3])
        @mpi_time times[4] transpose!(u[1], u[2])
        MEASURE_GATHER && @mpi_time times[5] gather(u[2])
    end

    @test u[1] == u_orig

    if myrank == 0
        print(
            """
            Processes:               $proc_dims
            Data dimensions:         $data_dims $(isempty(extra_dims) ? "" : "× $extra_dims")
            Permutations (1, 2, 3):  $(get_permutation.(pens))
            """)
        println("Transpositions (1 -> 2 -> 3 -> 2 -> 1):")
        println.(times[1:4])
        if MEASURE_GATHER
            println("Gather (config 2):")
            println(times[5])
        end
        println()
    end

    if PROFILE
        Profile.clear()
        @profile for it = 1:iterations
            transpose!(u[2], u[1])
            transpose!(u[3], u[2])
            transpose!(u[2], u[3])
            transpose!(u[1], u[2])
        end
        if myrank == 0
            open(io -> Profile.print(io, maxdepth=PROFILE_DEPTH),
                 PROFILE_OUTPUT, "w")
        end
    end

    nothing
end

function main()
    MPI.Init()

    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    if myrank == 0
        if Pencils.USE_ALLTOALLV
            @info "Using MPI_Alltoallv for transpositions"
        else
            @info "Using MPI_Isend / MPI_Irecv for transpositions"
        end
        println()
    end

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
    benchmark_decomp(comm, proc_dims, DIMS, with_permutations=Val(false))

    benchmark_decomp(comm, proc_dims, DIMS, extra_dims=(2, ),
                     iterations=ITERATIONS >> 1)
    benchmark_decomp(comm, proc_dims, DIMS, extra_dims=(2, ),
                     iterations=ITERATIONS >> 1, with_permutations=Val(false))

    MPI.Finalize()
end

main()
