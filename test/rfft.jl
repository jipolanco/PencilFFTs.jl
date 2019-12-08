#!/usr/bin/env julia

# Test 3D real-to-complex FFTs.

using PencilFFTs
import MPI

using BenchmarkTools
using LinearAlgebra
using Printf
using Random
using Test

include("include/FourierOperations.jl")
using .FourierOperations

const DATA_DIMS = (16, 16, 16)
const GEOMETRY = ((0.0, 4pi), (0.0, 2pi), (0.0, 2pi))

const DEV_NULL = @static Sys.iswindows() ? "nul" : "/dev/null"

# Compute and compare ⟨ |u|² ⟩ in physical and spectral space.
function test_global_average(u, uF, plan::PencilFFTPlan)
    comm = get_comm(plan)
    scale = get_scale_factor(plan)

    sum_u2_local = sqnorm(u)
    sum_uF2_local = sqnorm(uF)

    Ngrid = prod(size_global(pencil(u[1])))

    avg_u2 = MPI.Allreduce(sum_u2_local, +, comm) / Ngrid

    # To get a physically meaningful quantity, squared values in Fourier space
    # must be normalised by `scale` (and their sum is normalised again by
    # `scale` if one wants the average).
    # Equivalently, uF should be normalised by `sqrt(scale)`.
    @test scale == Ngrid
    avg_uF2 = MPI.Allreduce(sum_uF2_local, +, comm) / (Ngrid * Float64(scale))

    @test avg_u2 ≈ avg_uF2

    nothing
end

# Squared 2-norm of a tuple of arrays using LinearAlgebra.norm.
norm2(x::Tuple) = sum(norm.(x).^2)

function micro_benchmarks(u, uF, gF_global::FourierGrid)
    ωF = similar.(uF)
    gF_local = LocalGrid(gF_global, uF[1])

    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

    println("Micro-benchmarks:")

    @printf " - %-20s" "divergence global_view..."
    @btime divergence($uF, $gF_global)

    @printf " - %-20s" "divergence local..."
    @btime divergence($uF, $gF_local)

    @printf " - %-20s" "curl! global_view..."
    @btime curl!($ωF, $uF, $gF_global)

    ωF_copy = copy.(ωF)

    @printf " - %-20s" "curl! local..."
    @btime curl!($ωF, $uF, $gF_local)

    @test all(ωF .≈ ωF_copy)

    # For these, a generic implementation is used (LinearAlgebra.generic_norm2).
    @printf " - %-20s" "norm2(u)..."
    @btime norm2($u)

    @printf " - %-20s" "norm2(uF)..."
    @btime norm2($uF)

    # These are much faster because parent(u) is a regular Array, and Julia
    # calls BLAS in this case (LinearAlgebra.BLAS.nrm2).
    @printf " - %-20s" "norm2(parent(u))..."
    @btime norm2($(parent.(u)))

    @printf " - %-20s" "norm2(parent(uF))..."
    @btime norm2($(parent.(uF)))

    # Interestingly, this is even faster than BLAS!
    @printf " - %-20s" "sqnorm(u)..."
    @btime sqnorm($u)

    @printf " - %-20s" "sqnorm(parent(u))..."
    @btime sqnorm($(parent.(u)))

    @printf " - %-20s" "sqnorm(uF)..."
    @btime sqnorm($uF)

    nothing
end

function init_random_field!(u::PencilArray{T}, rng) where {T <: Complex}
    fill!(u, zero(T))

    u_g = global_view(u)
    perm = get_permutation(u)

    dims_global = size_global(pencil(u))
    ind_space = CartesianIndices(dims_global)
    ind_space_local = CartesianIndices(range_local(pencil(u), permute=false))
    @assert ndims(ind_space_local) == ndims(ind_space)

    scale = sqrt(2 * prod(dims_global))  # to get order-1 average values

    # Zero-based index of last element of r2c transform (which is set to zero)
    imax = dims_global[1] - 1

    # Loop over global dimensions, so that all processes generate the same
    # random numbers.
    for I in ind_space
        val = scale * randn(rng, T)

        I0 = Tuple(I) .- 1

        # First logical dimension, along which a r2c transform is applied.
        # If zero, Hermitian symmetry must be enforced.
        # (For this, zero-based indices are clearer!)
        i = I0[1]

        # Leave last element of r2c transform as zero.
        # Note: if I don't do this, the norms in physical and Fourier space
        # don't match... This is also the case if I set a real value to these
        # modes.
        i == imax && continue

        # We add in case a previous value was set by the code in the block
        # below (for Hermitian symmetry).
        if I ∈ ind_space_local
            Ip = Pencils.permute_indices(I, perm)
            u_g[Ip] += val
        end

        # If kx != 0, we're done.
        i == 0 || continue

        # Case kx == 0: account for Hermitian symmetry.
        #
        #    u(0, -ky, -kz) = conj(u(0, ky, kz))
        #
        # This also ensures that the zero mode is real.
        J0 = map((i, N) -> i == 0 ? 0 : N - i, I0, dims_global)
        J = CartesianIndex(J0 .+ 1)
        if J ∈ ind_space_local
            Jp = Pencils.permute_indices(J, perm)
            u_g[Jp] += conj(val)
        end
    end

    u
end

function main()
    MPI.Init()

    size_in = DATA_DIMS
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    pdims_2d = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    plan = PencilFFTPlan(size_in, Transforms.RFFT(), pdims_2d, comm)

    # Allocate and initialise vector field in Fourier space.
    uF = allocate_output(plan, Val(3))
    rng = MersenneTwister(42)
    init_random_field!.(uF, (rng, ))

    u = allocate_input(plan, Val(3))
    ldiv!.(u, plan, uF)
    test_global_average(u, uF, plan)

    @test sqnorm(u) ≈ norm2(u)

    # These are not the same because `FourierOperations.sqnorm` takes Hermitian
    # symmetry into account, so the result can be roughly twice as large.
    @test 1 < sqnorm(uF) / norm2(uF) <= 2 + 1e-8

    gF = FourierGrid(GEOMETRY, size_in, get_permutation(uF[1]))
    rank == 0 && micro_benchmarks(u, uF, gF)
    MPI.Barrier(comm)

    MPI.Finalize()
end

main()
