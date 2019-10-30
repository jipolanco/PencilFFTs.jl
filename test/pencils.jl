#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test

# TODO
# - test slab decomposition

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
    comm = get_comm(u_local)
    root = 0
    myrank = MPI.Comm_rank(comm)

    u = gather(u_local, root)
    v = gather(v_local, root)

    same = Ref(false)
    if u !== nothing && v !== nothing
        @assert myrank == root
        same[] = u == v
    end
    MPI.Bcast!(same, length(same), root, comm)

    same[]
end

function main()
    MPI.Init()

    Nxyz = (16, 21, 41)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    rng = MersenneTwister(42 + myrank)

    # Let MPI_Dims_create choose the values of (P1, P2).
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    topo = Pencils.Topology(comm, proc_dims)

    pen1 = Pencil(topo, Nxyz, (2, 3))
    pen2 = Pencil(pen1, (1, 3), permute=(2, 1, 3))
    pen3 = Pencil(pen2, (1, 2), permute=(3, 2, 1))

    test_array_wrappers(pen2, Float32)
    test_array_wrappers(pen3, Float64)

    @test Pencils.complete_dims(Val(5), (2, 3), (42, 12)) === (1, 42, 12, 1, 1)
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

    @assert Pencils.size_local(pen1) ==
        Pencils.size_remote(pen1, pen1.topology.coords_local...)

    u1 = PencilArray(pen1)
    u2 = PencilArray(pen2)
    u3 = PencilArray(pen3)

    # Set initial random data.
    randn!(rng, u1)
    u1 .+= 10 * myrank
    u1_orig = copy(u1)

    # Direct u1 -> u3 transposition is not possible!
    @test_throws ArgumentError transpose!(u3, u1)

    # Transpose back and forth between different pencil configurations
    transpose!(u2, u1)
    @test compare_distributed_arrays(u1, u2)

    transpose!(u3, u2)
    @test compare_distributed_arrays(u2, u3)

    transpose!(u2, u3)
    @test compare_distributed_arrays(u2, u3)

    transpose!(u1, u2)
    @test compare_distributed_arrays(u1, u2)

    @test u1_orig == u1

    # Test transpositions without permutations.
    let pen2 = Pencil(pen1, (1, 3))
        u2 = PencilArray(pen2)
        transpose!(u2, u1)
        @test compare_distributed_arrays(u1, u2)
    end

    # Test arrays with extra dimensions.
    let u1 = PencilArray(pen1, 3, 4)
        u2 = PencilArray(pen2, 3, 4)
        u3 = PencilArray(pen3, 3, 4)
        randn!(rng, u1)
        transpose!(u2, u1)
        @test compare_distributed_arrays(u1, u2)
        transpose!(u3, u2)
        @test compare_distributed_arrays(u2, u3)
    end

    # Test slab (1D) decomposition.
    let topo = Topology(comm, (Nproc, ))
        pen1 = Pencil(topo, Nxyz, (1, ))
        pen2 = Pencil(pen1, (2, ))
        u1 = PencilArray(pen1)
        u2 = PencilArray(pen2)
        randn!(rng, u1)
        transpose!(u2, u1)
        @test compare_distributed_arrays(u1, u2)
    end

    if Nproc == 1
        # @code_warntype Pencils.create_subcomms(Val(2), comm)
        # @code_warntype Pencils.Topology{2}(comm)
        # @code_warntype Pencils.get_cart_ranks_subcomm(pen1.topology.subcomms[1])

        # @code_warntype Pencils.to_local(pen2, (1, 2, 3))
        # @code_warntype Pencils.to_local(pen2, (1:2, 1:2, 1:2))

        # @code_warntype Pencils.size_local(pen2)

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
        # @code_warntype Pencils.transpose_impl!(1, u2, u1)
        # @code_warntype Pencils._get_remote_indices(1, (2, 3), 8)
    end

    MPI.Finalize()
end

main()
