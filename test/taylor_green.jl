#!/usr/bin/env julia

# Tests with a 3D Taylor-Green velocity field.
# https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex

using PencilFFTs

using MPI

include("include/Grids.jl")
using .Grids

const SAVE_VTK = false

if SAVE_VTK
    using WriteVTK
end

const DATA_DIMS = (64, 40, 32)
const GEOMETRY = ((0.0, 2pi), (0.0, 2pi), (0.0, 2pi))

const TG_U0 = 3.0
const TG_K0 = 2.0

const VectorField{T} = NTuple{3, PencilArray{T,3}} where T

function taylor_green!(u_local::VectorField, g::Grid, u0=TG_U0, k0=TG_K0)
    u = global_view.(u_local)

    for I in CartesianIndices(u[1])
        x, y, z = g[I]
        @inbounds u[1][I] = u0 * sin(k0 * x) * cos(k0 * y) * cos(k0 * z)
        @inbounds u[2][I] = -u0 * cos(k0 * x) * sin(k0 * y) * cos(k0 * z)
        @inbounds u[3][I] = 0
    end

    u_local
end

function field_to_vtk(g::Grid, u::VectorField, basename, fieldname)
    p = pencil(u[1])
    g_pencil = g[p]  # local geometry
    vtk_grid(basename, (g_pencil[1], g_pencil[2], g_pencil[3])) do vtk
        # This will only work if the data is not permuted!
        @assert get_permutation(p) === nothing
        vtk_point_data(vtk, u, fieldname)
    end
end

function divergence(uF_local::VectorField{T}, gk::FourierGrid, comm) where T
    uF = global_view.(uF_local)

    div = zero(T)

    for K in CartesianIndices(uF[1])
        # kx, ky, kz = gk[K]
        # div += (kx * uF[1][K] + ky * uF[2][K] + kz * uF[3][K]) * 1im
        k = Tuple(K)
        for n in eachindex(k)
            div += 1im * k[n] * uF[n][K]
        end
    end

    div_global = MPI.Allreduce(Ref(div), +, comm)

    div_global[]
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
    u = ntuple(_ -> allocate_input(plan), Val(3))  # allocate vector field

    g = Grid(GEOMETRY, size_in)
    taylor_green!(u, g)  # initialise TG velocity field

    uF = plan .* u  # apply 3D FFT
    # div = divergence()

    if SAVE_VTK
        field_to_vtk(g, u, "TG_proc_$(rank + 1)of$(Nproc)", "u")
    end

    MPI.Finalize()
end

main()
