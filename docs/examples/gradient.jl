# # Gradient of a scalar field
#
# This example shows different methods to compute the gradient of a real-valued
# 3D scalar field ``θ(\bm{x})`` in Fourier space, where $\bm{x} = (x, y, z)$.
# It is assumed that the field is periodic with period $L = 2π$ along all
# dimensions.
#
# ## General procedure
#
# The discrete Fourier expansion of ``θ`` writes
# ```math
# θ(\bm{x}) = ∑_{\bm{k} ∈ \Z^3} \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
# ```
# where $\bm{k} = (k_x, k_y, k_z)$ are the Fourier wave numbers and $\hat{θ}$ is
# the discrete Fourier transform of $θ$.
# Then, the spatial derivatives of $θ$ are given by
# ```math
# \frac{∂ θ(\bm{x})}{∂ x_i} =
# ∑_{\bm{k} ∈ \Z^3} i k_i \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
# ```
# where the subscript $i$ denotes one of the spatial components $x$, $y$ or
# $z$.
#
# In other words, to compute $\bm{∇} θ = (∂_x θ, ∂_y θ, ∂_z θ)$, one has to:
# 1. transform $θ$ to Fourier space to obtain $\hat{θ}$,
# 2. multiply $\hat{θ}$ by $i \bm{k}$,
# 3. transform the result back to physical space to obtain $\bm{∇} θ$.

# ## Preparation

# In this section, we initialise a random real-valued scalar field $θ$ and compute
# its FFT.
# For more details see the [Tutorial](@ref).

using MPI
using PencilFFTs
using Random

MPI.Init()

## Input data dimensions (Nx × Ny × Nz)
dims = (64, 32, 64)

## Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

## Automatically create decomposition configuration
comm = MPI.COMM_WORLD
pen = Pencil(dims, comm)

## Create plan
plan = PencilFFTPlan(pen, transform)

## Allocate data and initialise field
θ = allocate_input(plan)
randn!(θ)

## Perform distributed FFT
θ_hat = plan * θ
nothing  # hide

# Finally, we initialise the output that will hold ∇θ in Fourier space.
# Noting that ∇θ is a vector field, we choose to store it as a tuple of
# 3 PencilArrays.

∇θ_hat = allocate_output(plan, Val(3))

## This is equivalent:
## ∇θ_hat = ntuple(d -> similar(θ_hat), Val(3))

summary(∇θ_hat)

## Fourier wave numbers

# In general, the Fourier wave numbers are of the form
# ``k_i = 0, ±\frac{2π}{L_i}, ±\frac{4π}{L_i}, ±\frac{6π}{L_i}, …``,
# where ``L_i`` is the period along dimension ``i``.
# When a real-to-complex Fourier transform is applied, roughly half of
# these wave numbers are redundant due to the Hermitian symmetry of the complex
# Fourier coefficients.
# In practice, this means that for the fastest dimension $x$ (along which
# a real-to-complex transform is performed), the negative wave numbers are
# dropped, i.e. ``k_x = 0, \frac{2π}{L_x}, \frac{4π}{L_x}, …``.

# The `AbstractFFTs` package provides a convenient way to generate the Fourier
# wave numbers, using the functions
# [`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq)
# and
# [`rfftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfftfreq).
# We can use these functions to initialise a "grid" of wave numbers associated to
# our 3D real-to-complex transform:

using AbstractFFTs: fftfreq, rfftfreq

box_size = (2π, 2π, 2π)  # Lx, Ly, Lz
sample_rate = 2π .* dims ./ box_size

## In our case (Lx = 2π and Nx even), this gives kx = [0, 1, 2, ..., Nx/2].
kx = rfftfreq(dims[1], sample_rate[1])

## In our case (Ly = 2π and Ny even), this gives
## ky = [0, 1, 2, ..., Ny/2-1, -Ny/2, -Ny/2+1, ..., -1] (and similarly for kz).
ky = fftfreq(dims[2], sample_rate[2])
kz = fftfreq(dims[3], sample_rate[3])

kvec = (kx, ky, kz)

# Note that `kvec` now contains the wave numbers associated to the global domain.
# In the following, we will only need the wave numbers associated to the portion
# of the domain handled by the local MPI process.

# ## [Method 1: global views](@id gradient_method_global)

# [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray)s, returned for instance by [`allocate_input`](@ref)
# and  [`allocate_output`](@ref), take indices that start at 1, regardless of the
# location of the subdomain associated to the local process on the global grid.
# (In other words, `PencilArray`s take *local* indices.)
# On the other hand, we have defined the wave number vector `kvec` which,
# for each MPI process, is defined over the global domain, and as such it takes
# *global* indices.

# One straightforward way of making data arrays compatible with wave numbers is
# to use global views, i.e. arrays that take global indices.
# These are generated from `PencilArray`s by calling the [`global_view`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.global_view-Tuple{PencilArray})
# function.
# Note that, in general, global indices do *not* start at 1 for a given MPI
# process.
# A given process will own a range of data given by indices in `(i1:i2, j1:j2,
# k1:k2)`.

θ_glob = global_view(θ_hat)
∇θ_glob = global_view.(∇θ_hat)
summary(θ_glob)

# Once we have global views, we can combine data and wave numbers using the
# portion of global indices owned by the local MPI process, as shown below.
# We can use `CartesianIndices` to iterate over the global indices associated to
# the local process.

for I in CartesianIndices(θ_glob)
    i, j, k = Tuple(I)  # unpack indices

    ## Wave number vector associated to current Cartesian index.
    local kx, ky, kz  # hide
    kx = kvec[1][i]
    ky = kvec[2][j]
    kz = kvec[3][k]

    ## Compute gradient in Fourier space.
    ## Note that modifying ∇θ_glob also modifies the original PencilArray ∇θ_hat.
    ∇θ_glob[1][I] = im * kx * θ_glob[I]
    ∇θ_glob[2][I] = im * ky * θ_glob[I]
    ∇θ_glob[3][I] = im * kz * θ_glob[I]
end

# The above loop can be written in a slightly more efficient manner by precomputing
# `im * θ_glob[I]`:

@inbounds for I in CartesianIndices(θ_glob)
    i, j, k = Tuple(I)

    local kx, ky, kz  # hide
    kx = kvec[1][i]
    ky = kvec[2][j]
    kz = kvec[3][k]

    u = im * θ_glob[I]

    ∇θ_glob[1][I] = kx * u
    ∇θ_glob[2][I] = ky * u
    ∇θ_glob[3][I] = kz * u
end

# Also note that the above can be easily written in a more generic way, e.g. for
# arbitrary dimensions, thanks in part to the use of `CartesianIndices`.
# Moreover, in the above there is no notion of the dimension permutations
# discussed in [the tutorial](@ref tutorial:output_data_layout), as it is all
# hidden behind the implementation of `PencilArray`s.
# And as seen later in the [benchmarks](@ref gradient_benchmarks),
# these (hidden) permutations have zero cost, as the speed is identical
# to that of a function that explicitly takes into account these permutations.

# Finally, we can perform a backwards transform to obtain $\bm{∇} θ$ in physical
# space:

∇θ = plan \ ∇θ_hat;

# Note that the transform is automatically broadcast over the three fields
# of the `∇θ_hat` vector, and the result `∇θ` is also a tuple of
# three `PencilArray`s.

# ## [Method 2: explicit global indexing](@id gradient_method_global_explicit)

# Sometimes, one does not need to write generic code.
# In our case, one often knows the dimensionality of the problem and the
# memory layout of the data (i.e. the underlying index permutation).

# Below is a reimplementation of the above loop, using explicit indices instead
# of `CartesianIndices`, and assuming that the underlying index permutation is
# `(3, 2, 1)`, that is, data is stored in $(z, y, x)$ order.
# As discussed in [the tutorial](@ref tutorial:output_data_layout),
# this is the default for transformed arrays.
# This example also serves as a more explicit explanation for what is going on
# in the [first method](@ref gradient_method_global).

## Get local data range in the global grid.
rng = axes(θ_glob)  # = (i1:i2, j1:j2, k1:k2)

# For the loop below, we're assuming that the permutation is (3, 2, 1).
# In other words, the fastest index is the *last* one, and not the first one as
# it is usually in Julia.
# If the permutation is not (3, 2, 1), things will still work (well, except for
# the assertion below!), but the loop order will not be optimal.

@assert permutation(θ_hat) === Permutation(3, 2, 1)

@inbounds for i in rng[1], j in rng[2], k in rng[3]
    local kx, ky, kz  # hide
    kx = kvec[1][i]
    ky = kvec[2][j]
    kz = kvec[3][k]

    ## Note that we still access the arrays in (i, j, k) order.
    ## (The permutation happens behind the scenes!)
    u = im * θ_glob[i, j, k]

    ∇θ_glob[1][i, j, k] = kx * u
    ∇θ_glob[2][i, j, k] = ky * u
    ∇θ_glob[3][i, j, k] = kz * u
end

# ## [Method 3: using local indices](@id gradient_method_local)

# Alternatively, we can avoid global views and work directly on `PencilArray`s
# using local indices that start at 1.
# In this case, part of the strategy is to construct a "local" grid of wave
# numbers that can also be accessed with local indices.
# This can be conveniently done using the
# [`localgrid`](https://jipolanco.github.io/PencilArrays.jl/dev/LocalGrids/#PencilArrays.LocalGrids.localgrid)
# function of the PencilArrays.jl package, which accepts a `PencilArray` (or
# its associated `Pencil`) and the global coordinates (here `kvec`):

grid_fourier = localgrid(θ_hat, kvec)

# Note that one can directly iterate on the returned grid object:

@inbounds for I in CartesianIndices(grid_fourier)
    ## Wave number vector associated to current Cartesian index.
    local k⃗  # hide
    k⃗ = grid_fourier[I]
    u = im * θ_hat[I]
    ∇θ_hat[1][I] = k⃗[1] * u
    ∇θ_hat[2][I] = k⃗[2] * u
    ∇θ_hat[3][I] = k⃗[3] * u
end

# This implementation is as efficient as the other examples, while being
# slightly shorter to write.
# Moreover, it is quite generic, and can be made independent of the number of
# dimensions with little effort.

# ## Method 4: using broadcasting

# Finally, note that the local grid object returned by `localgrid` makes it is
# possible to compute the gradient using broadcasting, thus fully avoiding scalar
# indexing.
# This can be quite convenient in some cases, and can also be very useful if
# one is working on GPUs (where scalar indexing is prohibitively expensive).
# Using broadcasting, the above examples simply become:

@. ∇θ_hat[1] = im * grid_fourier[1] * θ_hat
@. ∇θ_hat[2] = im * grid_fourier[2] * θ_hat
@. ∇θ_hat[3] = im * grid_fourier[3] * θ_hat
nothing  # hide

# Once again, as shown in the [benchmarks](@ref gradient_benchmarks) further
# below, this method performs quite similarly to the other ones.

# ## Summary

# The `PencilArrays` module provides different alternatives to deal with
# MPI-distributed data that may be subject to dimension permutations.
# In particular, one can choose to work with *global* indices (first two
# examples) or with *local* indices (third example).

# If one wants to stay generic, making sure that the same code will work for
# arbitrary dimensions and will be efficient regardless of the underlying
# dimension permutation, methods [1](@ref gradient_method_global) and [3](@ref
# gradient_method_local) should be preferred.
# These use `CartesianIndices` and make no assumptions on the permutations
# (actually, permutations are completely invisible in the implementations).

# The [second method](@ref gradient_method_global_explicit) uses explicit
# `(i, j, k)` indices.
# It assumes that the underlying permutation is `(3, 2, 1)` to loop with `i` as
# the *slowest* index and `k` as the *fastest*, which is the optimal order in
# this case given the permutation.
# As such, the implementation is less generic than the others, and
# differences in performance are negligible with respect to more generic variants.

# ## [Benchmark results](@id gradient_benchmarks)

# The following are the benchmark results obtained from running
# [`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl)
# on a laptop, using 2 MPI processes and Julia 1.7.2, with an input array of
# global dimensions ``64 × 32 × 64``.
# The different methods detailed above are marked on the right.
# The "lazy" marks indicate runs where the wave numbers were represented by
# lazy `Frequencies` objects (returned by `rfftfreq` and `fftfreq`). Otherwise,
# they were collected into `Vector`s.
# For some reason, plain `Vector`s are faster when working with grids generated
# by `localgrid`.

# In the script, additional implementations can be found which rely on a more
# advanced understanding of permutations and on the internals of the
# [`PencilArrays`](https://jipolanco.github.io/PencilArrays.jl/dev/) package.
# For instance, `gradient_local_parent!` directly works with the raw
# data stored in Julia `Array`s, while `gradient_local_linear!` completely
# avoids `CartesianIndices` while staying generic and efficient.
# Nevertheless, these display roughly the same performance as the above examples.
#
#         gradient_global_view!...                  89.900 μs
#         gradient_global_view! (lazy)...           92.060 μs  [Method 1]
#         gradient_global_view_explicit!...         88.958 μs
#         gradient_global_view_explicit! (lazy)...  81.055 μs  [Method 2]
#         gradient_local!...                        92.305 μs
#         gradient_grid!...                         92.770 μs
#         gradient_grid! (lazy)...                  101.388 μs  [Method 3]
#         gradient_grid_broadcast!...               88.606 μs
#         gradient_grid_broadcast! (lazy)...        151.020 μs  [Method 4]
#         gradient_local_parent!...                 92.248 μs
#         gradient_local_linear!...                 91.212 μs
#         gradient_local_linear_explicit!...        90.992 μs
