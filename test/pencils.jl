#!/usr/bin/env julia

using PencilFFTs.Pencils

using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test

const DEV_NULL = @static Sys.iswindows() ? "nul" : "/dev/null"

function test_array_wrappers(p::Pencil)
    T = eltype(p)
    u = PencilArray(p)

    @test eltype(u) === eltype(u.data) === T

    let v = similar(u)
        @test typeof(v) === typeof(u)
        @test size(v) === size(u) === size(u.data)
    end

    let psize = size_local(p)
        a = zeros(T, psize)
        u = PencilArray(p, a)
        @test u.data === a
        @test IndexStyle(typeof(u)) === IndexStyle(typeof(a)) === IndexLinear()

        b = zeros(T, psize .+ 2)
        @test_throws DimensionMismatch PencilArray(p, b)
        @test_throws DimensionMismatch PencilArray(p, zeros(T, psize..., 3))

        # This is allowed.
        w = PencilArray(p, zeros(T, 3, psize...))
        @test size_global(w) === (3, size_global(p)...)

        @inferred PencilArray(p, zeros(T, 3, psize...))
        @inferred size_global(w)
    end

    let offsets = (3, 4)
        dims = (8, 3)
        x = randn(dims...)
        u = ShiftedArrayView(x, offsets)
        @test IndexStyle(typeof(u)) === IndexStyle(typeof(x)) === IndexLinear()
        @test axes(u) == (4:11, 5:7)
        @test u[6, 6] == x[3, 2]
        @test u[2] == x[2]  # linear indexing stays the same
        @test sum(u) ≈ sum(x)
        @test has_indices(u, 5, 6)
        @test !has_indices(u, 2, 1)

        s = similar(u, (4, 5))
        @test s isa ShiftedArrayView && size(s) == (4, 5)

        v = copy(u)
        @test v isa ShiftedArrayView && axes(v) === axes(u)
    end

    let offsets = (3, )
        x = randn(8)
        u = ShiftedArrayView(x, offsets)
        @test IndexStyle(typeof(u)) === IndexCartesian()  # special case in 1D
        @test axes(u) == (4:11, )
        @test u[6] == x[3]  # linear indexing is shifted (1D arrays only)
        @test sum(u) ≈ sum(x)
        @test has_indices(u, 5) && !has_indices(u, -2)
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
    root = 0

    myrank == root || redirect_stdout(open(DEV_NULL, "w"))

    rng = MersenneTwister(42 + myrank)

    # Let MPI_Dims_create choose the values of (P1, P2).
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    topo = MPITopology(comm, proc_dims)

    pen1 = Pencil(topo, Nxyz, (2, 3))
    pen2 = Pencil(pen1, decomp_dims=(1, 3), permute=(2, 1, 3))
    pen3 = Pencil(pen2, decomp_dims=(1, 2), permute=(3, 2, 1))

    @testset "Pencil constructor checks" begin
        # Too many decomposed directions
        @test_throws ArgumentError Pencil(
            MPITopology(comm, (Nproc, 1, 1)), Nxyz, (1, 2, 3))

        # Invalid permutation
        @test_throws ArgumentError Pencil(
            topo, Nxyz, (1, 2), permute=(0, 3, 15))

        # Decomposed dimensions may not be repeated.
        @test_throws ArgumentError Pencil(topo, Nxyz, (2, 2))

        # Decomposed dimensions must be in 1:N = 1:3.
        @test_throws ArgumentError Pencil(topo, Nxyz, (1, 4))
        @test_throws ArgumentError Pencil(topo, Nxyz, (0, 2))
    end

    @testset "PencilArray" begin
        test_array_wrappers(Pencil(pen2, Float32))
        test_array_wrappers(Pencil(pen3, Float64))
    end

    @testset "auxiliary functions" begin
        @test Pencils.complete_dims(Val(5), (2, 3), (42, 12)) ===
            (1, 42, 12, 1, 1)
        @test get_permutation(pen1) === nothing
        @test get_permutation(pen2) === (2, 1, 3)

        @test Pencils.relative_permutation(pen2, pen3) === (3, 1, 2)

        let a = (2, 1, 3), b = (3, 2, 1)
            @test Pencils.permute_indices((:a, :b, :c), (2, 3, 1)) ===
                (:b, :c, :a)
            a2b = Pencils.relative_permutation(a, b)
            @test Pencils.permute_indices(a, a2b) === b

            x = (3, 1, 2)
            x2nothing = Pencils.relative_permutation(x, nothing)
            @test Pencils.permute_indices(x, x2nothing) === (1, 2, 3)
        end
    end

    transpose_methods = (TransposeMethods.IsendIrecv(),
                         TransposeMethods.Alltoallv())

    @testset "transpose! $method" for method in transpose_methods
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
        let pen2 = Pencil(pen1, decomp_dims=(1, 3))
            u2 = PencilArray(pen2)
            transpose!(u2, u1)
            @test compare_distributed_arrays(u1, u2)
        end

    end

    # Test arrays with extra dimensions.
    @testset "extra dimensions" begin
        u1 = PencilArray(pen1, 3, 4)
        u2 = PencilArray(pen2, 3, 4)
        u3 = PencilArray(pen3, 3, 4)
        randn!(rng, u1)
        transpose!(u2, u1)
        @test compare_distributed_arrays(u1, u2)
        transpose!(u3, u2)
        @test compare_distributed_arrays(u2, u3)

        @inferred global_view(u1)
        let g = global_view(u1)
            @inferred axes(g)
            @inferred axes(g, 3)
        end
    end

    # Test slab (1D) decomposition.
    @testset "1D decomposition" begin
        topo = MPITopology(comm, (Nproc, ))
        pen1 = Pencil(topo, Nxyz, (1, ))
        pen2 = Pencil(pen1, decomp_dims=(2, ))
        u1 = PencilArray(pen1)
        u2 = PencilArray(pen2)
        randn!(rng, u1)
        transpose!(u2, u1)
        @test compare_distributed_arrays(u1, u2)

        # Same decomposed dimension as pen2, but different permutation.
        let pen = Pencil(pen2, decomp_dims=(2, ), permute=(3, 2, 1))
            v = PencilArray(pen)
            transpose!(v, u2)
            @test compare_distributed_arrays(u1, v)
        end

        # Test transposition between two identical configurations.
        transpose!(u2, u2)
        @test compare_distributed_arrays(u1, u2)

        let v = similar(u2)
            @test u2.pencil === v.pencil
            transpose!(v, u2)
            @test compare_distributed_arrays(u1, v)
        end
    end

    begin
        MPITopologies = Pencils.MPITopologies
        periods = zeros(Int, length(proc_dims))
        comm_cart = MPI.Cart_create(comm, collect(proc_dims), periods, false)
        @inferred MPITopologies.create_subcomms(Val(2), comm_cart)
        @inferred Pencils.MPITopology{2}(comm_cart)
        @inferred MPITopologies.get_cart_ranks_subcomm(pen1.topology.subcomms[1])

        @inferred Pencils.to_local(pen2, (1, 2, 3))
        @inferred Pencils.to_local(pen2, (1:2, 1:2, 1:2))

        @inferred Pencils.size_local(pen2)

        @inferred PencilArray(pen2)
        @inferred PencilArray(pen2, 3, 4)

        @inferred Pencils.permute_indices(Nxyz, (2, 3, 1))
        @inferred Pencils.relative_permutation((1, 2, 3), (2, 3, 1))
        @inferred Pencils.relative_permutation((1, 2, 3), nothing)

        u1 = PencilArray(pen1)
        u2 = PencilArray(pen2)

        @inferred Nothing gather(u2)
        @inferred transpose!(u2, u1)
        @inferred Pencils.transpose_impl!(1, u2, u1)
        @inferred Pencils._get_remote_indices(1, (2, 3), 8)
    end

    MPI.Finalize()
end

main()
