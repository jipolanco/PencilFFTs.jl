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

    pen1 = Pencil{1}(comm, Nxyz)
    pen2 = Pencil{2}(pen1, permute=(2, 1, 3))
    pen3 = Pencil{3}(pen1, permute=(3, 2, 1))

    @test index_permutation(pen1) === nothing
    @test index_permutation(pen2) === (2, 1, 3)

    @test Pencils.relative_permutation(pen2, pen3) === (3, 1, 2)

    let a = (2, 1, 3), b = (3, 2, 1)
        @test Pencils.permute_indices((:a, :b, :c), (2, 3, 1)) === (:b, :c, :a)
        a2b = Pencils.relative_permutation(a, b)
        @test Pencils.permute_indices(a, a2b) === b

        x = (3, 1, 2)
        x2nothing = Pencils.relative_permutation(x, nothing)
        @test Pencils.permute_indices(x, x2nothing) === (1, 2, 3)
    end

    if myrank == 0
        @show pen1.axes_all
        @show pen2.axes_all
        @show pen3.axes_all
    end

    @test Pencils.put_colon(Val(1), (2, 3, 4)) === (:, 3, 4)
    @test Pencils.put_colon(Val(3), (2, 3, 4)) === (2, 3, :)

    @assert Pencils.size_local(pen1) ==
        Pencils.size_remote(pen1, pen1.topology.coords_local...)

    u1 = allocate(pen1)
    u2 = allocate(pen2)

    transpose!(u2, pen2, u1, pen1)

    if Nproc == 1
        # @code_warntype Pencils.create_subcomms(Val(2), comm)
        # @code_warntype Pencils.Topology{2}(comm)
        # @code_warntype Pencils.get_cart_ranks_subcomm(pen1.topology.subcomms[1])

        # @code_warntype Pencils.to_local(pen2, (1, 2, 3))
        # @code_warntype Pencils.to_local(pen2, (1:2, 1:2, 1:2))

        # @code_warntype Pencils.put_colon(Val(1), (2, 3, 4))

        # @code_warntype Pencils.size_remote(pen1, 1, 1)
        # @code_warntype Pencils.size_remote(pen1, 1, :)
        # @code_warntype Pencils.size_remote(pen1, :, 1)
        # @code_warntype Pencils.size_remote(pen1, :, :)

        # @code_warntype Pencils.permute_indices(Nxyz, (2, 3, 1))
        # @code_warntype Pencils.relative_permutation((1, 2, 3), (2, 3, 1))
        # @code_warntype Pencils.relative_permutation((1, 2, 3), nothing)

        @code_warntype Pencils.transpose_impl!(Val(1), u2, pen2, u1, pen1)
    end

    MPI.Finalize()
end

main()
