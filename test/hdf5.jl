#!/usr/bin/env julia

# Packages must be loaded in this order!
using MPI
using HDF5
using PencilFFTs.PencilArrays

using Random
using Test

include("include/MPITools.jl")
using .MPITools

const FILENAME_H5 = "fields.h5"

function test_write(u::PencilArray)
    comm = get_comm(u)
    info = MPI.Info()

    v = copy(u)
    w = copy(u)
    v .+= 1
    w .+= 2

    @test_nowarn h5open(FILENAME_H5, "w", "fapl_mpio", (comm, info)) do ff
        @test isopen(ff)
        @test_nowarn ff["scalar", collective=true, chunks=false] = u
        @test_nowarn ff["vector", collective=false, chunks=true] = (u, v, w)
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
    pen = Pencil(topo, Nxyz, (1, 3), permute=Val((2, 3, 1)))
    u = PencilArray(pen)
    randn!(rng, u)
    u .+= 10 * myrank

    @testset "write HDF5" begin
        test_write(u)
    end

    HDF5.h5_close()
    MPI.Finalize()
end

main()
