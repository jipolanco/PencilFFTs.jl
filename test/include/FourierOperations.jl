module FourierOperations

export divergence, curl!, sqnorm

# TODO
# - Can I make this module independent of Pencils?

using PencilFFTs.Pencils
using Reexport

include("Grids.jl")
@reexport using .Grids

const VectorField{T} = NTuple{3, PencilArray{T,3}}

"""
    divergence(u::AbstractArray{<:Complex}, grid::FourierGrid)

Compute total divergence ``∑|∇⋅u|²`` in Fourier space, in the local process.
"""
function divergence(uF_local::VectorField{T},
                    grid::FourierGrid) where {T <: Complex}
    div2 = real(zero(T))

    uF = map(global_view, uF_local)
    ux = first(uF)

    @inbounds for I in CartesianIndices(ux)
        k = grid[I]  # (kx, ky, kz)
        l = LinearIndices(ux)[I]
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
                    grid::LocalFourierGrid) where {T <: Complex}
    div2 = real(zero(T))
    @inbounds for i in eachindex(grid, uF...)::Base.OneTo
        k = grid[i]  # (kx, ky, kz)
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
    u = global_view.(uF_local)
    ω = global_view.(ωF_local)

    # TODO improve Grid to be more consistent with array indices.
    # It should have similar axes, and ideally linear indexing.

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
               grid::LocalFourierGrid) where {T <: Complex}
    @inbounds for I in eachindex(grid) :: Base.OneTo
        k = grid[I]  # (kx, ky, kz)
        v = (u[1][I], u[2][I], u[3][I])
        ω[1][I] = 1im * (k[2] * v[3] - k[3] * v[2])
        ω[2][I] = 1im * (k[3] * v[1] - k[1] * v[3])
        ω[3][I] = 1im * (k[1] * v[2] - k[2] * v[1])
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
index_r2c(u::PencilArray) = index_r2c(get_permutation(u))
index_r2c(::Nothing) = 1
index_r2c(::Val{p}) where {p} = findfirst(==(1), p) :: Int

"""
    sqnorm(u::AbstractArray{<:Complex})

Compute squared norm of array in Fourier space, in the local process.
"""
function sqnorm(u::PencilArray{T}) where {T <: Complex}
    s = zero(real(T))
    g = global_view(u)

    ind = index_r2c(u)

    for I in CartesianIndices(g)
        # Account for Hermitian symmetry implied by r2c transform along the
        # first logical dimension.
        i_r2c = Tuple(I)[ind]
        factor = i_r2c == 1 ? 1 : 2
        s += factor * abs2(g[I])
    end

    s
end

# Add a variant for real arrays, for completeness.
sqnorm(u::AbstractArray{T} where {T <: Real}) = sum(abs2, u)

sqnorm(u::Tuple) = mapreduce(sqnorm, +, u)

end
