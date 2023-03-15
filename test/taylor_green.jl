#!/usr/bin/env julia

# Tests with a 3D Taylor-Green velocity field.
# https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex

using PencilFFTs

import MPI

using BenchmarkTools
using Test

include("include/FourierOperations.jl")
using .FourierOperations
import .FourierOperations: VectorField

include("include/MPITools.jl")
using .MPITools

const SAVE_VTK = false

if SAVE_VTK
    using WriteVTK
end

const DATA_DIMS = (16, 8, 16)
const GEOMETRY = ((0.0, 4pi), (0.0, 2pi), (0.0, 2pi))

const TG_U0 = 3.0
const TG_K0 = 2.0

# Initialise TG flow (global view version).
function taylor_green!(u_local::VectorField, g::PhysicalGrid, u0=TG_U0, k0=TG_K0)
    u = map(global_view, u_local)

    @inbounds for I in CartesianIndices(u[1])
        x, y, z = g[I]
        u[1][I] =  u0 * sin(k0 * x) * cos(k0 * y) * cos(k0 * z)
        u[2][I] = -u0 * cos(k0 * x) * sin(k0 * y) * cos(k0 * z)
        u[3][I] = 0
    end

    u_local
end

# Initialise TG flow (local grid version).
function taylor_green!(u::VectorField, g::PhysicalGridIterator, u0=TG_U0, k0=TG_K0)
    @assert size_local(u[1]) === size(g)

    @inbounds for (i, (x, y, z)) in enumerate(g)
        u[1][i] =  u0 * sin(k0 * x) * cos(k0 * y) * cos(k0 * z)
        u[2][i] = -u0 * cos(k0 * x) * sin(k0 * y) * cos(k0 * z)
        u[3][i] = 0
    end

    u
end

# Verify vorticity of Taylor-Green flow.
function check_vorticity_TG(ω::VectorField{T}, g::PhysicalGridIterator, comm,
                            u0=TG_U0, k0=TG_K0) where {T}
    diff2 = zero(T)

    @inbounds for (i, (x, y, z)) in enumerate(g)
        ω_TG = (
            -u0 * k0 * cos(k0 * x) * sin(k0 * y) * sin(k0 * z),
            -u0 * k0 * sin(k0 * x) * cos(k0 * y) * sin(k0 * z),
            2u0 * k0 * sin(k0 * x) * sin(k0 * y) * cos(k0 * z),
        )
        for n = 1:3
            diff2 += (ω[n][i] - ω_TG[n])^2
        end
    end

    MPI.Allreduce(diff2, +, comm)
end

function fields_to_vtk(g::PhysicalGridIterator, basename; fields...)
    isempty(fields) && return

    # This works but it's heavier, since g.data is a dense array:
    # xyz = g.data
    # It would generate a structured grid (.vts) file, instead of rectilinear
    # (.vtr).

    xyz = ntuple(n -> g.grid[n][g.range[n]], Val(3))

    vtk_grid(basename, xyz) do vtk
        for (name, u) in pairs(fields)
            vtk[string(name)] = u
        end
    end
end

MPI.Init()

size_in = DATA_DIMS
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

silence_stdout(comm)

pdims_2d = let pdims_in = zeros(Int, 2)
    pdims = MPI.Dims_create(Nproc, pdims_in)
    ntuple(i -> Int(pdims[i]), 2)
end

plan = PencilFFTPlan(
    size_in, Transforms.RFFT(), pdims_2d, comm;
    permute_dims=Val(true),
)
u = allocate_input(plan, Val(3))  # allocate vector field

g_global = PhysicalGrid(GEOMETRY, size_in, permutation(u))
g_local = LocalGridIterator(g_global, u)
taylor_green!(u, g_local)   # initialise TG velocity field

uF = plan * u  # apply 3D FFT

gF_global = FourierGrid(GEOMETRY, size_in, permutation(uF))
gF_local = LocalGridIterator(gF_global, uF)
ωF = similar.(uF)

@testset "Taylor-Green" begin
    let u_glob = similar.(u)
        # Compare with initialisation using global indices
        taylor_green!(u_glob, g_global)
        @test all(u .≈ u_glob)
    end

    div2 = divergence(uF, gF_local)

    # Compare local and global versions of divergence
    @test div2 == divergence(uF, gF_global)

    div2_mean = MPI.Allreduce(div2, +, comm) / prod(size_in)
    @test div2_mean ≈ 0 atol=1e-16

    curl!(ωF, uF, gF_local)
    ω = plan \ ωF

    # Test global version of curl
    ωF_glob = similar.(ωF)
    curl!(ωF_glob, uF, gF_global)
    @test all(ωF_glob .== ωF)

    ω_err = check_vorticity_TG(ω, g_local, comm)
    @test ω_err ≈ 0 atol=1e-16
end

# Micro-benchmarks
print("divergence local...  ")
@btime divergence($uF, $gF_local)

print("divergence global... ")
@btime divergence($uF, $gF_global)

print("curl! local...       ")
@btime curl!($ωF, $uF, $gF_local)

print("curl! global...      ")
ωF_glob = similar.(ωF)
@btime curl!($ωF_glob, $uF, $gF_global)

if SAVE_VTK
    fields_to_vtk(
        g_local, "TG_proc_$(rank + 1)of$(Nproc)";
        u = u, ω = ω,
    )
end
