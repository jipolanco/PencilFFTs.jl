module FourierOperations

export divergence, curl!, norm2

# TODO
# - Can I make this module independent of Pencils?
#   -> I need `Grid`s to be local, and a standard alternative to spatial_indices
#      (CartesianIndices? eachindex?)

using PencilFFTs.Pencils
using Reexport

include("Grids.jl")
@reexport using .Grids

"""
    divergence(u::AbstractArray{<:Complex}, grid::FourierGrid)

Compute total divergence ``∑|∇⋅u|²`` in Fourier space, in the local process.
"""
function divergence(uF_local::PencilArray{T},
                    grid::FourierGrid) where {T <: Complex}
    uF = global_view(uF_local)
    div2 = real(zero(T))

    @inbounds for I in spatial_indices(uF)
        k = grid[I]  # (kx, ky, kz)
        div = zero(T)
        for n in eachindex(k)
            v = 1im * k[n] * uF[I, n]
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
function curl!(ωF_local::AbstractArray{T,N},
               uF_local::AbstractArray{T,N},
               grid::FourierGrid) where {T <: Complex, N}
    u = global_view(uF_local)
    ω = global_view(ωF_local)

    @inbounds for I in spatial_indices(u)
        k = grid[I]  # (kx, ky, kz)
        v = (u[I, 1], u[I, 2], u[I, 3])
        ω[I, 1] = 1im * (k[2] * v[3] - k[3] * v[2])
        ω[I, 2] = 1im * (k[3] * v[1] - k[1] * v[3])
        ω[I, 3] = 1im * (k[1] * v[2] - k[2] * v[1])
    end

    ωF_local
end

"""
    norm2(u::AbstractArray{<:Complex})

Compute squared norm of array in Fourier space, in the local process.
"""
function norm2(u::PencilArray{T}) where {T <: Complex}
    s = zero(real(T))
    g = global_view(u)

    # We account for Hermitian symmetry implied by the r2c transform along the
    # first *logical* dimension. Due to the index permutation in `u`, the r2c
    # dimension is not necessarily the first one in `u`.
    # TODO I should make this easier...
    index_r2c = let perm = get_permutation(u)
        if perm === nothing
            1
        else
            findfirst(==(1), Pencils.extract(perm)) :: Int
        end
    end

    for I in CartesianIndices(g)
        # Account for Hermitian symmetry implied by r2c transform along the
        # first logical dimension.
        i_r2c = Tuple(I)[index_r2c]
        factor = i_r2c == 1 ? 1 : 2
        s += factor * abs2(g[I])
    end

    s
end

# Add a variant for real arrays, for completeness.
norm2(u::AbstractArray{T} where {T <: Real}) = sum(abs2, u)

end
