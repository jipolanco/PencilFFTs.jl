#!/usr/bin/env julia

# Packages must be loaded in this order!
using MPI
using HDF5
using PencilFFTs.PencilArrays
using PencilFFTs.PencilIO

if !PencilIO.hdf5_has_parallel()
    @warn "HDF5 has no parallel support. Skipping HDF5 tests."
    exit(0)
end

using Random
using Test

include("include/MPITools.jl")
using .MPITools

function test_write(filename, u::PencilArray)
    comm = get_comm(u)
    info = MPI.Info()
    rank = MPI.Comm_rank(comm)

    v = copy(u)
    w = copy(u)
    v .+= 1
    w .+= 2

    # Open file in serial mode first.
    if rank == 0
        h5open(filename, "w") do ff
            # "HDF5 file was not opened with the MPIO driver"
            @test_throws ErrorException ff["scalar"] = u
        end
    end

    MPI.Barrier(comm)

    @test_nowarn phdf5_open(filename, "w", comm, info) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar", collective=true, chunks=false] = u
        @test_nowarn ff["vector_tuple", collective=false, chunks=true] = (u, v, w)
        @test_nowarn ff["vector_array", collective=true, chunks=true] = [u, v, w]
    end

    # TODO
    # - read back data
    # - check that components (u, v, w) are written as expected
    @test_nowarn phdf5_open(filename, "r", comm, info) do ff
        @test isopen(ff)
    end

    nothing
end

function main()
    MPI.Init()

    Nxyz = (16, 21, 41)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    silence_stdout(comm)

    # Let MPI_Dims_create choose the values of (P1, P2).
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    rng = MersenneTwister(42)

    topo = MPITopology(comm, proc_dims)
    pen = Pencil(topo, Nxyz, (1, 3), permute=Permutation(2, 3, 1))
    u = PencilArray(pen)
    randn!(rng, u)
    u .+= 10 * myrank

    filename = MPI.bcast(tempname(), 0, comm)

    @testset "write HDF5" begin
        test_write(filename, u)
    end

    HDF5.h5_close()
    MPI.Finalize()
end

main()
