#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI

using InteractiveUtils
using Test

function main()
    MPI.Init()

    Nxyz = (16, 31, 13)
    Nproc = MPI.Comm_size(MPI.COMM_WORLD)
    myrank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Let MPI_Dims_create choose the values of (P1, P2).
    P1, P2 = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    dims = [P1, P2]
    periods = [0, 0]
    reorder = false
    comm = MPI.Cart_create(MPI.COMM_WORLD, dims, periods, reorder)

    # @code_warntype Pencils.get_cart_ranks_matrix(Val(2), comm)
    # @code_warntype Pencils.create_subcomms(Val(2), comm)
    # @code_warntype Pencils.Topology{2}(comm)

    pen1 = Pencil{1}(comm, Nxyz)

    pen1_bis = Pencil{1}(pen1)
    @test pen1 === pen1_bis

    pen2 = Pencil{2}(pen1)
    pen3 = Pencil{3}(pen1)

    if myrank == 0
        @show pen1.axes_all
        @show pen2.axes_all
        @show pen3.axes_all
    end

    u1 = allocate(pen1)
    u2 = allocate(pen2)

    transpose!(u2, pen2, u1, pen1)

    MPI.Finalize()
end

main()
