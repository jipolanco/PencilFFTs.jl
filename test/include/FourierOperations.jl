module FourierOperations

export divergence, curl!, sqnorm

using PencilFFTs.PencilArrays
using Reexport

include("Grids.jl")
@reexport using .Grids

const VectorField{T} = NTuple{N, PencilArray{T,3}} where {N}

"""
    divergence(u::AbstractArray{<:Complex}, grid::FourierGrid)

Compute total divergence ``∑|∇⋅u|²`` in Fourier space, in the local process.
"""
function divergence(uF_local::VectorField{T},
                    grid::FourierGrid) where {T <: Complex}
    div2 = real(zero(T))

    uF = map(global_view, uF_local)
    ux = first(uF)

    @inbounds for (l, I) in enumerate(CartesianIndices(ux))
        k = grid[I]  # (kx, ky, kz)
        div = zero(T)
        for n in eachindex(k)
            v = 1im * k[n] * uF[n][l]
            div += v
        end
        div2 += abs2(div)
    end

    div2
end

# Local grid variant (faster -- with linear indexing!)
function divergence(uF::VectorField{T},
                    grid::FourierGridIterator) where {T <: Complex}
    div2 = real(zero(T))
    @inbounds for (i, k) in enumerate(grid)
        div = zero(T)
        for n in eachindex(k)
            v = 1im * k[n] * uF[n][i]
            div += v
        end
        div2 += abs2(div)
    end
    div2
end

"""
    curl!(ω, u, grid::FourierGrid)

Compute ``ω = ∇×u`` in Fourier space.
"""
function curl!(ωF_local::VectorField{T},
               uF_local::VectorField{T},
               grid::FourierGrid) where {T <: Complex}
    u = map(global_view, uF_local)
    ω = map(global_view, ωF_local)

    @inbounds for I in CartesianIndices(u[1])
        k = grid[I]  # (kx, ky, kz)
        l = LinearIndices(u[1])[I]
        v = (u[1][l], u[2][l], u[3][l])
        ω[1][l] = 1im * (k[2] * v[3] - k[3] * v[2])
        ω[2][l] = 1im * (k[3] * v[1] - k[1] * v[3])
        ω[3][l] = 1im * (k[1] * v[2] - k[2] * v[1])
    end

    ωF_local
end

function curl!(ω::VectorField{T},
               u::VectorField{T},
               grid::FourierGridIterator) where {T <: Complex}
    @inbounds for (i, k) in enumerate(grid)
        v = (u[1][i], u[2][i], u[3][i])
        ω[1][i] = 1im * (k[2] * v[3] - k[3] * v[2])
        ω[2][i] = 1im * (k[3] * v[1] - k[1] * v[3])
        ω[3][i] = 1im * (k[1] * v[2] - k[2] * v[1])
    end
    ω
end

"""
    index_r2c(u::PencilArray)

Return index associated to dimension of real-to-complex transform.

This is assumed to be the first *logical* dimension of the array.
Since indices in the array may be permuted, the actual dimension may be other
than the first.
"""
index_r2c(u::PencilArray) = index_r2c(permutation(u))
index_r2c(::Nothing) = 1
index_r2c(::Val{p}) where {p} = findfirst(==(1), p) :: Int

"""
    sqnorm(u::AbstractArray{<:Complex}, grid::FourierGridIterator)

Compute squared norm of array in Fourier space, in the local process.
"""
sqnorm(u::PencilArray, grid) = sqnorm((u, ), grid)

function sqnorm(u::VectorField{T}, grid::FourierGridIterator) where {T <: Complex}
    gp = parent(grid) :: FourierGrid
    kx = gp[1]  # global wave numbers along r2c dimension

    # Note: when Nx (size of real input data) is even, the Nyquist frequency is
    # also counted twice.
    Nx = size(gp, 1)
    @assert length(kx) == div(Nx, 2) + 1

    k_zero = kx[1]  # zero mode

    kx_lims = if iseven(Nx)
        (k_zero, kx[end])  # kx = 0 or Nx/2 (Nyquist frequency)
    else
        # We repeat k_zero for type inference reasons.
        (k_zero, k_zero)  # only kx = 0
    end

    s = zero(real(T))

    @inbounds for (i, k) in enumerate(grid)
        # Account for Hermitian symmetry implied by r2c transform along the
        # first logical dimension. Note that `k` is "unpermuted", meaning that
        # k[1] is the first *logical* wave number.
        factor = k[1] in kx_lims ? 1 : 2
        s += factor * sum(v -> abs2(v[i]), u)
    end

    s
end

# Add a variant for real arrays, for completeness.
sqnorm(u::AbstractArray{T} where {T <: Real}) = sum(abs2, u)

sqnorm(u::Tuple, args...) = mapreduce(v -> sqnorm(v, args...), +, u)

end
