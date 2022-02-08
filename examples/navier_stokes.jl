# # Navier--Stokes equations
#
# In this example, we numerically solve the incompressible Navier--Stokes
# equations
#
# ```math
# ∂_t \bm{v} + (\bm{v} ⋅ \bm{∇}) \bm{v} = -\frac{1}{ρ} \bm{∇} p + ν ∇^2 \bm{v},
# \quad \bm{∇} ⋅ \bm{v} = 0,
# ```
#
# where ``\bm{v}(\bm{x}, t)`` and ``p(\bm{x}, t)`` are respectively the velocity
# and pressure fields, ``ν`` is the fluid kinematic viscosity and ``ρ`` is the
# fluid density.
#
# We solve the above equations a 3D periodic domain using a standard Fourier
# pseudo-spectral method.

# ## First steps
#
# We start by loading the required packages, initialising MPI and setting the
# simulation parameters.

using MPI
using PencilFFTs

MPI.Init()
comm = MPI.COMM_WORLD

## Simulation parameters
Ns = (32, 32, 32)  # = (Nx, Ny, Nz)
Ls = (2π, 2π, 2π)  # = (Lx, Ly, Lz)

## Collocation points ("global" = over all processes).
## We include the endpoint (length = N + 1) for convenience.
xs_global = map((N, L) -> range(0, L; length = N + 1), Ns, Ls)  # = (x, y, z)

# Let's check the number of MPI processes over which we're running our
# simulation:
MPI.Comm_size(comm)

# We can now create a partitioning of the domain based on the number of grid
# points (`Ns`) and on the number of MPI processes.
# There are different ways to do this.
# For simplicity, here we do it automatically following the [PencilArrays.jl
# docs](https://jipolanco.github.io/PencilArrays.jl/stable/Pencils/#pencil-high-level):

pen = Pencil(Ns, comm)

# We construct a distributed vector field that follows this decomposition
# configuration:

v⃗₀ = (
    PencilArray{Float64}(undef, pen),  # vx
    PencilArray{Float64}(undef, pen),  # vy
    PencilArray{Float64}(undef, pen),  # vz
)
summary(v⃗₀[1])

# We still need to fill this array with interesting values that represent a
# physical velocity field.

# ## Initial condition

# Let's set the initial condition in physical space.
# In this example, we choose the [Taylor--Green
# vortex](https://en.wikipedia.org/wiki/Taylor%E2%80%93Green_vortex)
# configuration as an initial condition:
#
# ```math
# \begin{aligned}
# v_x(x, y, z) &=  u₀ \sin(k₀ x) \cos(k₀ y) \cos(k₀ z) \\
# v_y(x, y, z) &= -u₀ \cos(k₀ x) \sin(k₀ y) \cos(k₀ z) \\
# v_z(x, y, z) &= 0
# \end{aligned}
# ```
#
# where ``u₀`` and ``k₀`` are two parameters setting the amplitude and the
# period of the velocity field.
#
# To set the initial condition, each MPI process needs to know which portion of
# the physical grid it has been attributed.
# For this, PencilArrays.jl includes a
# [`localgrid`](https://jipolanco.github.io/PencilArrays.jl/dev/LocalGrids/#PencilArrays.LocalGrids.localgrid)
# helper function:

grid = localgrid(pen, xs_global)

# We can use this to initialise the velocity field:

u₀ = 1.0
k₀ = 2π / Ls[1]  # should be integer if L = 2π (to preserve periodicity)

@. v⃗₀[1] =  u₀ * sin(k₀ * grid.x) * cos(k₀ * grid.y) * cos(k₀ * grid.z)
@. v⃗₀[2] = -u₀ * cos(k₀ * grid.x) * sin(k₀ * grid.y) * cos(k₀ * grid.z)
@. v⃗₀[3] =  0
nothing # hide

# Let's plot a 2D slice of the velocity field managed by the local MPI process:

using GLMakie
using GLMakie.Makie.GeometryBasics

## Compute the norm of a vector field represented by a tuple of arrays.
function vecnorm(v⃗::NTuple)
    vnorm = similar(v⃗[1])
    for n ∈ eachindex(v⃗[1])
        w = zero(eltype(vnorm))
        for v ∈ v⃗
            w += v[n]^2
        end
        vnorm[n] = sqrt(w)
    end
    vnorm
end

let fig = Figure(resolution = (700, 600))
    ax = Axis3(fig[1, 1]; aspect = :data, xlabel = "x", ylabel = "y", zlabel = "z")
    vnorm = vecnorm(v⃗₀)
    xs_local = LocalGrids.components(grid)
    ## ct = contour!(
    ##     ax, xs_local..., vnorm;
    ##     alpha = 0.04, levels = 0:0.25:1,
    ##     colormap = :viridis, colorrange = extrema(vnorm),
    ## )
    ct = volume!(
        ax, xs_local..., vnorm;
        transparency = true,
    )
    ## Colorbar(fig[1, 2], ct; label = "Velocity magnitude")
    ## Plot global domain limits
    ## mesh!(ax, HyperRectangle(0, 0, 0, Ls[1], Ls[2], Ls[3]); strokewidth = 1, color = RGBAf(0, 0, 0, 0))
    fig
end

# ## Velocity in Fourier space
#
# In the Fourier pseudo-spectral method, the periodic velocity field is
# discretised in space as a truncated Fourier series
#
# ```math
# \bm{v}(\bm{x}, t) =
# ∑_{\bm{k}} \hat{\bm{v}}_{\bm{k}}(t) \, e^{i \bm{k} ⋅ \bm{x}},
# ```
#
# where ``\bm{k} = (k_x, k_y, k_z)`` are the discrete wave numbers.
#
# The wave numbers can be obtained using the
# [`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/dev/api/#AbstractFFTs.fftfreq)
# function.
# Since we perform a real-to-complex transform along the first dimension, we use
# [`rfftfreq`](https://juliamath.github.io/AbstractFFTs.jl/dev/api/#AbstractFFTs.rfftfreq) instead for ``k_x``:

using AbstractFFTs: fftfreq, rfftfreq

ks_global = (
    rfftfreq(Ns[1], 2π * Ns[1] / Ls[1]),  # kx | real-to-complex
     fftfreq(Ns[2], 2π * Ns[2] / Ls[2]),  # ky | complex-to-complex
     fftfreq(Ns[3], 2π * Ns[3] / Ls[3]),  # kz | complex-to-complex
)

ks_global[1]'
#
ks_global[2]'
#
ks_global[3]'

# To transform the velocity field to Fourier space, we first create a
# real-to-complex FFT plan to be applied to one of the velocity components:

plan = PencilFFTPlan(v⃗₀[1], Transforms.RFFT())

# See [`PencilFFTPlan`](@ref) for details on creating plans and on optional
# keyword arguments.
#
# We can now apply this plan to the three velocity components to obtain the
# respective Fourier coefficients ``\hat{\bm{v}}_{\bm{k}}``:

v̂s = plan .* v⃗₀
summary(v̂s[1])

# Note that, in Fourier space, the domain decomposition is performed along the
# directions ``x`` and ``y``:

pencil(v̂s[1])

# This is because the 3D FFTs are performed one dimension at a time, with the
# ``x`` direction first and the ``z`` direction last.
# To efficiently perform an FFT along a given direction (taking advantage of
# serial FFT implementations like FFTW), all the data along that direction must
# be contained locally within a single MPI process.
# For that reason, data redistributions (or *transpositions*) among MPI
# processes are performed behind the scenes during each FFT computation.
# Such transpositions require important communications between MPI processes,
# and are usually the most time-consuming aspect of massively-parallel
# simulations using this kind of methods.
#
# To solve the Navier--Stokes equations in Fourier space, we will also need the
# respective wave numbers ``\bm{k}`` associated to the local MPI process.
# Similarly to the local grid points, these are obtained using the `localgrid`
# function:

grid_fourier = localgrid(v̂s[1], ks_global)

# ## Computing the non-linear term
#
# One can show that, in Fourier space, the incompressible Navier--Stokes
# equations can be written as
#
# ```math
# ∂_t \hat{\bm{v}}_{\bm{k}} =
# - \mathcal{P}_{\bm{k}} \! \left[ \widehat{(\bm{v} ⋅ \bm{∇}) \bm{v}} \right]
# - ν |\bm{k}|^2 \hat{\bm{v}}_{\bm{k}}
# \quad \text{ with } \quad
# \mathcal{P}_{\bm{k}}(\hat{\bm{F}}_{\bm{k}}) = \left( I - \frac{\bm{k} ⊗
# \bm{k}}{|\bm{k}|^2} \right) \hat{\bm{F}}_{\bm{k}},
# ```
#
# where ``\mathcal{P}_{\bm{k}}`` is a projection operator allowing to preserve the
# incompressibility condition ``\bm{∇} ⋅ \bm{v} = 0``, and ``I`` is the identity
# matrix.
#
# Now that we have the wave numbers ``\bm{k}``, computing the linear viscous
# term in Fourier space is straighforward once we have the Fourier coefficients
# ``\hat{\bm{v}}_{\bm{k}}`` of the velocity field.
# What is slightly more challenging (and much more costly) is the computation of
# the non-linear term in Fourier space, ``\hat{\bm{F}}_{\bm{k}} =
# \left[ \widehat{(\bm{v} ⋅ \bm{∇}) \bm{v}} \right]_{\bm{k}}``.
# Following the pseudo-spectral method, the quadratic nonlinearity is computed
# by collocation in physical space, while derivatives are computed in Fourier
# space.
# This requires transforming fields back and forth between both spaces.
#
# Below we implement a function that computes the non-linear term in Fourier
# space based on its convective form ``(\bm{v} ⋅ \bm{∇}) \bm{v} = \bm{∇} ⋅
# (\bm{v} ⊗ \bm{v})``.
# This equivalence uses the incompressibility condition ``\bm{∇} ⋅ \bm{v} = 0``.

using LinearAlgebra: mul!, ldiv!  # for applying FFT plans in-place

## Compute non-linear term in Fourier space from velocity field in physical
## space. Optional keyword arguments may be passed to avoid allocations.
function ns_nonlinear!(
        F̂s, vs, plan, grid_fourier;
        vbuf = similar(vs[1]), v̂buf = similar(F̂s[1]),
    )
    ## Compute F_i = ∂_j (v_i v_j) for each i.
    ## In Fourier space: F̂_i = im * k_j * FFT(v_i * v_j)
    w, ŵ = vbuf, v̂buf
    @inbounds for (i, F̂i) ∈ enumerate(F̂s)
        F̂i .= 0
        vi = vs[i]
        for (j, vj) ∈ enumerate(vs)
            w .= vi .* vj     # w = v_i * v_j in physical space
            mul!(ŵ, plan, w)  # same in Fourier space
            ## Add derivative in Fourier space
            for I ∈ eachindex(grid_fourier)
                k⃗ = grid_fourier[I]  # = (kx, ky, kz)
                kj = k⃗[j]
                F̂i[I] += im * kj * ŵ[I]
            end
        end
    end
    F̂s
end

# As an example, we may use this function on our initial velocity field:

F̂s = similar.(v̂s)
ns_nonlinear!(F̂s, v⃗₀, plan, grid_fourier);

# Strictly speaking, computing the non-linear term by collocation can lead to
# [aliasing
# errors](https://en.wikipedia.org/wiki/Aliasing#Sampling_sinusoidal_functions),
# as the quadratic term generates Fourier modes that fall beyond the range of
# resolved wave numbers.
# The typical solution is to apply Orzsag's 2/3 rule to zero-out the Fourier
# coefficients associated to the highest wave numbers.
# We define a function that applies this procedure below.

function dealias_twothirds!(ŵs::Tuple, grid_fourier, ks_global)
    ks_max = maximum.(abs, ks_global)  # maximum stored wave numbers (kx_max, ky_max, kz_max)
    ks_lim = (2 / 3) .* ks_max
    @inbounds for I ∈ eachindex(grid_fourier)
        k⃗ = grid_fourier[I]
        if any(abs.(k⃗) .> ks_lim)
            for ŵ ∈ ŵs
                ŵ[I] = 0
            end
        end
    end
    ŵs
end

## We can apply this on the previously computed non-linear term:
dealias_twothirds!(F̂s, grid_fourier, ks_global);

# Finally, we implement the projection associated to the incompressibility
# condition:

function project_divergence_free!(ŵs, ûs, grid_fourier)
    @inbounds for (i, ŵ) ∈ enumerate(ŵs)
        ûi = ûs[i]
        for I ∈ eachindex(grid_fourier)
            k⃗ = grid_fourier[I]  # = (kx, ky, kz)
            k² = sum(abs2, k⃗)
            iszero(k²) && continue  # avoid division by zero
            wtmp = ûi[I]
            for (j, ûj) ∈ enumerate(ûs)
                wtmp += k⃗[i] * k⃗[j] * ûj[I] / k²
            end
            ŵ[I] = wtmp
        end
    end
    ŵs
end

F̂s_divfree = similar.(F̂s)
project_divergence_free!(F̂s_divfree, F̂s, grid_fourier);

# We can verify the correctness of the projection operator by checking that the
# initial velocity field is not modified by it, since it is already
# incompressible:

v̂s_proj = project_divergence_free!(similar.(v̂s), v̂s, grid_fourier)
v̂s_proj .≈ v̂s

# ## Putting it all together
#
# To perform the time integration of the Navier--Stokes equations, we will use
# the timestepping routines implemented in the DifferentialEquations.jl suite.
# For simpicity, we will use an explicit Runge--Kutta scheme.
# In this case, we just need to write a function that computes the right-hand
# side of the Navier--Stokes equations in Fourier space:

function ns_rhs!(
        dv̂s::NTuple{N, <:PencilArray}, v̂s::NTuple{N, <:PencilArray}, p, t,
    ) where {N}
    ## 1. Transform velocity to physical space
    plan = p.plan
    cache = p.cache
    vs = cache.vs
    map((v, v̂) -> ldiv!(v, plan, v̂), vs, v̂s)

    ## 2. Compute non-linear term and dealias it
    grid_fourier = p.grid_fourier
    ks_global = p.ks_global
    F̂s = cache.F̂s
    ns_nonlinear!(F̂s, vs, plan, grid_fourier; vbuf = cache.vtmp, v̂buf = cache.v̂tmp)
    dealias_twothirds!(F̂s, grid_fourier, ks_global)

    ## 3. Project into divergence-free space
    project_divergence_free!(dv̂s, F̂s, grid_fourier)

    ## 4. Add viscous term (and multiply projected NL term by -1)
    ν = p.ν
    map(dv̂s, v̂s) do dv̂, v̂
        @inbounds for I ∈ eachindex(grid_fourier)
            k⃗ = grid_fourier[I]  # = (kx, ky, kz)
            k² = sum(abs2, k⃗)
            dv̂[I] = -dv̂[I] - im * ν * k² * v̂[I]
        end
    end

    nothing
end

# For the time-stepping, we load OrdinaryDiffEq.jl from the
# DifferentialEquations.jl suite and set-up the simulation.

using OrdinaryDiffEq
using StructArrays: StructArray, components
using StaticArrays: SVector

## Solver parameters and temporary variables
params = (;
    ν = 1.0,  # kinematic viscosity
    plan, grid_fourier, ks_global,
    cache = (
        vs = similar.(v⃗₀),
        F̂s = similar.(v̂s),
        vtmp = similar(v⃗₀[1]),
        v̂tmp = similar(v̂s[1]),
    )
)

tspan = (0.0, 1.0)
v̂s_init = plan .* v⃗₀

# Since DifferentialEquations.jl can't directly deal with tuples of arrays, we
# convert the input data to `StructArray{SVector{3, ComplexF64}}`.

to_structarray(vs::NTuple{N, A}) where {N, A <: AbstractArray} =
    StructArray{SVector{N, eltype(A)}}(vs)

v̂s_init_ode = to_structarray(v̂s_init)
summary(v̂s_init_ode)

# We also need to define a couple of interface functions to make things work
# with DifferentialEquations.

ns_rhs!(dv::StructArray, v::StructArray, args...) =
    ns_rhs!(components(dv), components(v), args...)

## This is also needed for the DifferentialEquations interface.
## TODO should this be defined in PencilArrays.jl?
Base.similar(us::StructArray{T, <:Any, <:Tuple{Vararg{PencilArray}}}) where {T} =
    StructArray{T}(map(similar, components(us))) :: typeof(us)

prob = ODEProblem(ns_rhs!, v̂s_init_ode, tspan, params)
integrator = init(prob, RK4(); dt = 1e-3, save_everystep = false);

# We finally solve the problem over time and plot the solution.

v⃗_plot = Observable(plan .\ components(integrator.u))
t_plot = Observable(integrator.t)

fig = let
    fig = Figure(resolution = (700, 600))
    ax = Axis(
        fig[1, 1]; title = @lift("t = $(round($t_plot, digits = 3))"),
        aspect = DataAspect(), xlabel = "x", ylabel = "y",
    )
    xs, ys = grid[1], grid[2]
    slice = (:, :, 1)
    v_slices = @lift view.($v⃗_plot, slice...)
    v_norm = @lift vec(@. sqrt($v_slices[1]^2 + $v_slices[2]^2 + $v_slices[3]^2))
    vx = @lift $v_slices[1]
    vy = @lift $v_slices[2]
    vz = @lift $v_slices[3]
    ar = arrows!(
        ax, xs, ys, vx, vy;
        arrowcolor = v_norm, linecolor = v_norm,
        arrowsize = 5, lengthscale = 0.3,
        colormap = :viridis, colorrange = (0.0, 1.0),
    )
    ## cb = contour!(
    ##     ax, xs, ys, vz;
    ##     colormap = :RdBu, levels = -0.025:0.01:0.025,
    ##     linewidth = 4,
    ##     extendlow = :auto, extendhigh = :auto,
    ## )
    Colorbar(fig[1, 2], ar; label = "Velocity magnitude")
    ## Plot global domain limits
    poly!(ax, Rect(0, 0, Ls[1], Ls[2]); strokewidth = 1, color = RGBAf(0, 0, 0, 0))
    fig
end

procid = MPI.Comm_rank(comm) + 1

record(fig, "velocity_proc$procid.mp4"; framerate = 10) do io
    while integrator.t ≤ 2.0
        dt = 0.01
        step!(integrator, dt)
        t_plot[] = integrator.t
        ldiv!.(v⃗_plot[], plan, components(integrator.u))
        v⃗_plot[] = v⃗_plot[]  # to force updating the plot
        recordframe!(io)
    end
end;

# #md ```@raw html
# #md <figure class="video_container">
# #md   <video controls="true" allowfullscreen="true">
# #md     <source src="velocity_proc1.mp4" type="video/mp4">
# #md   </video>
# #md </figure>
# #md ```
