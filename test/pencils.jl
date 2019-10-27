#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test

function test_array_wrappers(p::Pencil, ::Type{T}=Float32) where T
    u = PencilArray(p, T)

    @test eltype(u) === eltype(u.data) === T

    let v = similar(u)
        @test typeof(v) === typeof(u)
        @test size(v) === size(u) === size(u.data)
    end

    let psize = size_local(p)
        a = zeros(T, psize)
        u = PencilArray(p, a)
        @test u.data === a

        b = zeros(T, psize .+ 2)
        @test_throws DimensionMismatch PencilArray(p, b)
        @test_throws DimensionMismatch PencilArray(p, zeros(T, psize..., 3))

        # This is allowed.
        w = PencilArray(p, zeros(T, 3, psize...))
        @test size_global(w) === (3, size_global(p)...)

        # @code_warntype PencilArray(p, zeros(T, 3, psize...))
        # @code_warntype size_global(w)
    end

    nothing
end

function compare_distributed_arrays(u_local::PencilArray, v_local::PencilArray)
    u = gather(u_local)
    v = gather(v_local)

    if u === nothing || v === nothing
        return
    end

    @test u == v

    nothing
end

function main()
    MPI.Init()

    Nxyz = (160, 221, 203)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    rng = MersenneTwister(42 + myrank)

    # Let MPI_Dims_create choose the values of (P1, P2).
    P1, P2 = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    topo = Pencils.Topology(comm, (P1, P2))

    pen1 = Pencil{1}(topo, Nxyz)
    pen2 = Pencil{2}(pen1, permute=(2, 1, 3))
    pen3 = Pencil{3}(pen1, permute=(3, 2, 1))

    test_array_wrappers(pen2, Float32)
    test_array_wrappers(pen3, Float64)

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

    @test Pencils.put_colon(Val(1), (2, 3, 4)) === (:, 3, 4)
    @test Pencils.put_colon(Val(3), (2, 3, 4)) === (2, 3, :)

    @assert Pencils.size_local(pen1) ==
        Pencils.size_remote(pen1, pen1.topology.coords_local...)

    u1 = PencilArray(pen1)
    u2 = PencilArray(pen2)

    # Set initial random data.
    randn!(rng, u1)
    u1 .+= 10 * myrank

    transpose!(u2, u1)
    @time transpose!(u2, u1)

    let t0 = MPI.Wtime()
        transpose!(u2, u1)
        dt = MPI.Wtime() - t0
        MPI.Barrier(comm)
        myrank == 0 && @info "transpose: $dt seconds"
    end

    compare_distributed_arrays(u1, u2)

    if Nproc == 1
        # @code_warntype Pencils.create_subcomms(Val(2), comm)
        # @code_warntype Pencils.Topology{2}(comm)
        # @code_warntype Pencils.get_cart_ranks_subcomm(pen1.topology.subcomms[1])

        # @code_warntype Pencils.to_local(pen2, (1, 2, 3))
        # @code_warntype Pencils.to_local(pen2, (1:2, 1:2, 1:2))

        # @code_warntype Pencils.size_local(pen2)

        # @code_warntype Pencils.put_colon(Val(1), (2, 3, 4))

        # @code_warntype PencilArray(pen2)
        # @code_warntype PencilArray(pen2, 3, 4)
        # @code_warntype PencilArray(pen2, Float32, 3, 4)

        # @code_warntype Pencils.size_remote(pen1, 1, 1)
        # @code_warntype Pencils.size_remote(pen1, 1, :)
        # @code_warntype Pencils.size_remote(pen1, :, 1)
        # @code_warntype Pencils.size_remote(pen1, :, :)

        # @code_warntype Pencils.permute_indices(Nxyz, (2, 3, 1))
        # @code_warntype Pencils.relative_permutation((1, 2, 3), (2, 3, 1))
        # @code_warntype Pencils.relative_permutation((1, 2, 3), nothing)

        # @code_warntype gather(u2)

        # @code_warntype transpose!(u2, u1)
        # @code_warntype Pencils.transpose_impl!(Val(1), u2, pen2, u1, pen1)
    end

    MPI.Finalize()
end

main()
