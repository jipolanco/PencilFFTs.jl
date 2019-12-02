#!/usr/bin/env julia

# Tests with a 3D Taylor-Green velocity field.
# https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex

using PencilFFTs

import FFTW
import MPI

using BenchmarkTools
using InteractiveUtils
using LinearAlgebra
using Printf
using Profile
using Test

include("include/Grids.jl")
using .Grids

const SAVE_VTK = false

if SAVE_VTK
    using WriteVTK
end

const DATA_DIMS = (16, 8, 16)
const GEOMETRY = ((0.0, 4pi), (0.0, 2pi), (0.0, 2pi))

const TG_U0 = 3.0
const TG_K0 = 2.0

const DEV_NULL = @static Sys.iswindows() ? "nul" : "/dev/null"

const VectorField{T} = PencilArray{T,4}

function taylor_green!(u_local::VectorField, g::Grid, u0=TG_U0, k0=TG_K0)
    u = global_view(u_local)

    for I in spatial_indices(u)
        x, y, z = g[I]
        @inbounds u[I, 1] =  u0 * sin(k0 * x) * cos(k0 * y) * cos(k0 * z)
        @inbounds u[I, 2] = -u0 * cos(k0 * x) * sin(k0 * y) * cos(k0 * z)
        @inbounds u[I, 3] = 0
    end

    u_local
end

# Verify vorticity of Taylor-Green flow.
function check_vorticity_TG(ω_local::VectorField, g::Grid, comm,
                            u0=TG_U0, k0=TG_K0)
    ω = global_view(ω_local)
    diff2 = zero(eltype(ω))

    for I in spatial_indices(ω)
        x, y, z = g[I]
        ω_TG = (
            -u0 * k0 * cos(k0 * x) * sin(k0 * y) * sin(k0 * z),
            -u0 * k0 * sin(k0 * x) * cos(k0 * y) * sin(k0 * z),
            2u0 * k0 * sin(k0 * x) * sin(k0 * y) * cos(k0 * z),
        )
        for n = 1:3
            diff2 += (ω[I, n] - ω_TG[n])^2
        end
    end

    MPI.Allreduce(diff2, +, comm)
end

function fields_to_vtk(g::Grid, basename, fields::Vararg{Pair})
    isempty(fields) && return
    p = pencil(first(fields).second)
    g_pencil = g[p]  # local geometry
    xyz = ntuple(n -> g_pencil[n], 3)
    vtk_grid(basename, xyz) do vtk
        for p in fields
            name = p.first
            u = p.second
            v = ntuple(n -> @view(u[:, :, :, n]), 3)
            vtk_point_data(vtk, v, name)
        end
    end
end

# Compute total divergence ⟨|∇⋅u|²⟩ in Fourier space, in the local process.
function divergence(uF_local::VectorField{T},
                    gF::FourierGrid) where {T <: Complex}
    uF = global_view(uF_local)
    div2 = real(zero(T))

    @inbounds for I in spatial_indices(uF)
        K = gF[I]  # (kx, ky, kz)
        div = zero(T)
        for n in eachindex(K)
            v = 1im * K[n] * uF[I, n]
            div += v
        end
        div2 += abs2(div)
    end

    div2
end

# Compute ω = ∇×u in Fourier space.
function curl!(ωF_local::VectorField{T}, uF_local::VectorField{T},
               gF::FourierGrid) where {T <: Complex}
    u = global_view(uF_local)
    ω = global_view(ωF_local)

    @inbounds for I in spatial_indices(u)
        K = gF[I]  # (kx, ky, kz)
        v = (u[I, 1], u[I, 2], u[I, 3])
        ω[I, 1] = 1im * (K[2] * v[3] - K[3] * v[2])
        ω[I, 2] = 1im * (K[3] * v[1] - K[1] * v[3])
        ω[I, 3] = 1im * (K[1] * v[2] - K[2] * v[1])
    end

    ωF_local
end

mynorm(u) = sqrt(sum(abs2, u))

function micro_benchmarks(u, uF, gF)
    ωF = similar(uF)

    @test mynorm(u) ≈ norm(u)

    BenchmarkTools.DEFAULT_PARAMETERS.seconds = 1

    println("Micro-benchmarks:")

    @printf " - %-20s" "norm(u)..."
    @btime norm($u)

    @printf " - %-20s" "norm(uF)..."
    @btime norm($uF)

    @printf " - %-20s" "norm(parent(u))..."
    @btime norm($(parent(u)))

    @printf " - %-20s" "norm(parent(uF))..."
    @btime norm($(parent(uF)))

    @printf " - %-20s" "mynorm(u)..."
    @btime mynorm($u)

    @printf " - %-20s" "mynorm(uF)..."
    @btime mynorm($uF)

    @printf " - %-20s" "divergence..."
    @btime divergence($uF, $gF)

    @printf " - %-20s" "curl!..."
    @btime curl!($ωF, $uF, $gF)

    nothing
end

function main()
    MPI.Init()

    size_in = DATA_DIMS
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    rank = MPI.Comm_rank(comm)

    rank == 0 || redirect_stdout(open(DEV_NULL, "w"))

    pdims_2d = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    plan = PencilFFTPlan(size_in, Transforms.RFFT(), pdims_2d, comm,
                         extra_dims=(3, ), permute_dims=Val(true))
    u = allocate_input(plan)  # allocate vector field

    g = Grid(GEOMETRY, size_in, get_permutation(u))
    taylor_green!(u, g)  # initialise TG velocity field

    uF = plan * u  # apply 3D FFT
    gF = FourierGrid(GEOMETRY, size_in, get_permutation(uF))
    ωF = similar(uF)

    rank == 0 && micro_benchmarks(u, uF, gF)
    MPI.Barrier(comm)

    let div2 = divergence(uF, gF)
        div2_total = MPI.Allreduce(div2, +, comm)
        @test div2_total ≈ 0 atol=1e-16
    end

    curl!(ωF, uF, gF)
    ω = plan \ ωF

    ω_err = check_vorticity_TG(ω, g, comm)
    @test ω_err ≈ 0 atol=1e-16

    if SAVE_VTK
        fields_to_vtk(g, "TG_proc_$(rank + 1)of$(Nproc)",
                      "u" => u, "ω" => ω)
    end

    MPI.Finalize()
end

main()
