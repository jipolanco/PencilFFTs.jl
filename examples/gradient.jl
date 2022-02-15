# Different implementations of gradient computation in Fourier space, with
# performance comparisons.

# Some sample benchmark results (on Julia 1.4):
#
#     Transforms: (RFFT, FFT, FFT)
#     Input type: Float64
#     Global dimensions: (64, 32, 64)  ->  (33, 32, 64)
#     MPI topology: 2D decomposition (2×1 processes)
#
#     gradient_global_view!...            184.853 μs (0 allocations: 0 bytes)
#     gradient_global_view_explicit!...   124.993 μs (0 allocations: 0 bytes)
#     gradient_local!...                  146.369 μs (0 allocations: 0 bytes)
#     gradient_local_parent!...           145.743 μs (0 allocations: 0 bytes)
#     gradient_local_linear!...           145.679 μs (0 allocations: 0 bytes)
#     gradient_local_linear_explicit!...  145.543 μs (0 allocations: 0 bytes)
#
# This was obtained when running julia with the default optimisation level -O2.

using BenchmarkTools
using MPI
using PencilFFTs
using PencilFFTs.LocalGrids: LocalRectilinearGrid

using AbstractFFTs: fftfreq, rfftfreq
using Printf: @printf
using Random: randn!

const PA = PencilFFTs.PencilArrays

const INPUT_DIMS = (64, 32, 64)

function generate_wavenumbers_r2c(dims::Dims{3})
    box_size = (2π, 2π, 2π)  # Lx, Ly, Lz
    sample_rate = 2π .* dims ./ box_size

    # In our case (Lx = 2π and Nx even), this gives kx = [0, 1, 2, ..., Nx/2].
    kx = rfftfreq(dims[1], sample_rate[1])

    # In our case (Ly = 2π and Ny even), this gives
    # ky = [0, 1, 2, ..., Ny/2-1, -Ny/2, -Ny/2+1, ..., -1] (and similarly for kz).
    ky = fftfreq(dims[2], sample_rate[2])
    kz = fftfreq(dims[3], sample_rate[3])

    (kx, ky, kz)
end

# Compute and return ∇θ in Fourier space, using global views.
function gradient_global_view!(∇θ_hat::NTuple{3,PencilArray},
                               θ_hat::PencilArray, kvec_global)
    # Generate OffsetArrays that take global indices.
    θ_glob = global_view(θ_hat)
    ∇θ_glob = map(global_view, ∇θ_hat)

    @inbounds for (n, I) in enumerate(CartesianIndices(θ_glob))
        i, j, k = Tuple(I)  # global indices

        # Wave number vector associated to current Cartesian index.
        kx = kvec_global[1][i]
        ky = kvec_global[2][j]
        kz = kvec_global[3][k]

        u = im * θ_glob[n]

        ∇θ_glob[1][n] = kx * u
        ∇θ_glob[2][n] = ky * u
        ∇θ_glob[3][n] = kz * u
    end

    ∇θ_hat
end

function gradient_global_view_explicit!(∇θ_hat::NTuple{3,PencilArray},
                                        θ_hat::PencilArray, kvec_global)
    # Generate OffsetArrays that take global indices.
    θ_glob = global_view(θ_hat)
    ∇θ_glob = map(global_view, ∇θ_hat)

    rng = axes(θ_glob)  # (i1:i2, j1:j2, k1:k2)

    # Note: since the dimensions in Fourier space are permuted as (z, y, x), it
    # is faster to loop with `k` as the fastest index.
    @assert permutation(θ_hat) === Permutation(3, 2, 1)

    @inbounds for i in rng[1], j in rng[2], k in rng[3]
        # Wave number vector associated to current Cartesian index.
        kx = kvec_global[1][i]
        ky = kvec_global[2][j]
        kz = kvec_global[3][k]

        u = im * θ_glob[i, j, k]

        ∇θ_glob[1][i, j, k] = kx * u
        ∇θ_glob[2][i, j, k] = ky * u
        ∇θ_glob[3][i, j, k] = kz * u
    end

    ∇θ_hat
end

# Compute and return ∇θ in Fourier space, using local indices.
function gradient_local!(∇θ_hat::NTuple{3,PencilArray}, θ_hat::PencilArray,
                         kvec_local)
    @inbounds for (n, I) in enumerate(CartesianIndices(θ_hat))
        i, j, k = Tuple(I)  # local indices

        # Wave number vector associated to current Cartesian index.
        kx = kvec_local[1][i]
        ky = kvec_local[2][j]
        kz = kvec_local[3][k]

        u = im * θ_hat[n]

        ∇θ_hat[1][n] = kx * u
        ∇θ_hat[2][n] = ky * u
        ∇θ_hat[3][n] = kz * u
    end

    ∇θ_hat
end

function gradient_grid!(∇θ_hat, θ_hat, grid_fourier::LocalRectilinearGrid)
    @inbounds for I in CartesianIndices(grid_fourier)
        k⃗ = grid_fourier[I]
        u = im * θ_hat[I]
        ∇θ_hat[1][I] = k⃗[1] * u
        ∇θ_hat[2][I] = k⃗[2] * u
        ∇θ_hat[3][I] = k⃗[3] * u
    end
    ∇θ_hat
end

function gradient_grid_broadcast!(∇θ_hat, θ_hat, g::LocalRectilinearGrid)
    @. ∇θ_hat[1] = im * g[1] * θ_hat
    @. ∇θ_hat[2] = im * g[2] * θ_hat
    @. ∇θ_hat[3] = im * g[3] * θ_hat
    ∇θ_hat
end

# Compute and return ∇θ in Fourier space, using local indices on the raw data
# (which takes permuted indices).
function gradient_local_parent!(∇θ_hat::NTuple{3,PencilArray},
                                θ_hat::PencilArray, kvec_local)
    θ_p = parent(θ_hat) :: Array
    ∇θ_p = parent.(∇θ_hat)

    perm = permutation(θ_hat)

    @inbounds for (n, I) in enumerate(CartesianIndices(θ_p))
        # Unpermute indices to (i, j, k)
        J = perm \ I

        # Wave number vector associated to current Cartesian index.
        i, j, k = Tuple(J)  # local indices
        kx = kvec_local[1][i]
        ky = kvec_local[2][j]
        kz = kvec_local[3][k]

        u = im * θ_p[n]

        ∇θ_p[1][n] = kx * u
        ∇θ_p[2][n] = ky * u
        ∇θ_p[3][n] = kz * u
    end

    ∇θ_hat
end

# Similar to gradient_local!, but avoiding CartesianIndices.
function gradient_local_linear!(∇θ_hat::NTuple{3,PencilArray},
                                θ_hat::PencilArray, kvec_local)
    # We want to iterate over the arrays in memory order to maximise
    # performance. For this we need to take into account the permutation of
    # indices in the Fourier-transformed arrays. By default, the memory order in
    # Fourier space is (z, y, x) instead of (x, y, z), but this is never assumed
    # below. The wave numbers must be permuted accordingly.
    perm = permutation(θ_hat)  # e.g. Permutation(3, 2, 1)
    kvec_perm = perm * kvec_local  # e.g. (kz, ky, kx)

    # Create wave number iterator.
    kvec_iter = Iterators.product(kvec_perm...)

    @inbounds for (n, kvec_n) in enumerate(kvec_iter)
        # Apply inverse permutation to the current wave number vector.
        # Note that this permutation has zero cost, since perm is a
        # compile-time constant!
        # (This can be verified by comparing the performance of this function
        # with the "explicit" variant of `gradient_local_linear`, below.)
        κ = perm \ kvec_n  # = (kx, ky, kz)

        u = im * θ_hat[n]

        # Note that this is very easy to generalise to N dimensions...
        ∇θ_hat[1][n] = κ[1] * u
        ∇θ_hat[2][n] = κ[2] * u
        ∇θ_hat[3][n] = κ[3] * u
    end

    ∇θ_hat
end

# Less generic version of the above, assuming that the permutation is (3, 2, 1).
# It's basically the same but probably easier to understand.
function gradient_local_linear_explicit!(∇θ_hat::NTuple{3,PencilArray},
                                         θ_hat::PencilArray, kvec_local)
    @assert permutation(θ_hat) === Permutation(3, 2, 1)

    # Create wave number iterator in (kz, ky, kx) order, i.e. in the same order
    # as the array data.
    kvec_iter = Iterators.product(kvec_local[3], kvec_local[2], kvec_local[1])

    @inbounds for (n, kvec_n) in enumerate(kvec_iter)
        kz, ky, kx = kvec_n
        u = im * θ_hat[n]
        ∇θ_hat[1][n] = kx * u
        ∇θ_hat[2][n] = ky * u
        ∇θ_hat[3][n] = kz * u
    end

    ∇θ_hat
end

MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
dims = INPUT_DIMS

kvec = generate_wavenumbers_r2c(dims)  # as tuple of Frequencies
kvec_collected = collect.(kvec)        # as tuple of Vector

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# MPI topology information
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# Disable output on all but one process.
rank == 0 || redirect_stdout(devnull)

# Automatically create decomposition configuration
pen = Pencil(dims, comm)

# Create plan
plan = PencilFFTPlan(pen, transform)
println(plan, "\n")

# Allocate data and initialise field
θ = allocate_input(plan)
randn!(θ)

# Perform distributed FFT
θ_hat = plan * θ

# Local part of the grid in Fourier space
grid_fourier_lazy = localgrid(pencil(θ_hat), kvec)
grid_fourier_col = localgrid(pencil(θ_hat), collect.(kvec))

# Compute and compare gradients using different methods.
# Note that these return a tuple of 3 PencilArrays representing a vector
# field.
∇θ_hat_base = allocate_output(plan, Val(3))
∇θ_hat_other = similar.(∇θ_hat_base)

# Local wave numbers: (kx[i1:i2], ky[j1:j2], kz[k1:k2]).
kvec_local = getindex.(kvec, range_local(θ_hat))

gradient_global_view!(∇θ_hat_base, θ_hat, kvec)

@printf "%-40s" "gradient_global_view!..."
@btime gradient_global_view!($∇θ_hat_other, $θ_hat, $kvec_collected)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_global_view! (lazy)..."
@btime gradient_global_view!($∇θ_hat_other, $θ_hat, $kvec)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_global_view_explicit!..."
@btime gradient_global_view_explicit!($∇θ_hat_other, $θ_hat, $kvec_collected)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_global_view_explicit! (lazy)..."
@btime gradient_global_view_explicit!($∇θ_hat_other, $θ_hat, $kvec)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_local!..."
@btime gradient_local!($∇θ_hat_other, $θ_hat, $kvec_local);
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_grid!..."
@btime gradient_grid!($∇θ_hat_other, $θ_hat, $grid_fourier_col);
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_grid! (lazy)..."
@btime gradient_grid!($∇θ_hat_other, $θ_hat, $grid_fourier_lazy);
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_grid_broadcast!..."
@btime gradient_grid_broadcast!($∇θ_hat_other, $θ_hat, $grid_fourier_col);
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_grid_broadcast! (lazy)..."
@btime gradient_grid_broadcast!($∇θ_hat_other, $θ_hat, $grid_fourier_lazy);
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_local_parent!..."
@btime gradient_local_parent!($∇θ_hat_other, $θ_hat, $kvec_local)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_local_linear!..."
@btime gradient_local_linear!($∇θ_hat_other, $θ_hat, $kvec_local)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

@printf "%-40s" "gradient_local_linear_explicit!..."
@btime gradient_local_linear_explicit!($∇θ_hat_other, $θ_hat, $kvec_local)
@assert all(∇θ_hat_base .≈ ∇θ_hat_other)

# Get gradient in physical space.
∇θ = plan \ ∇θ_hat_base
