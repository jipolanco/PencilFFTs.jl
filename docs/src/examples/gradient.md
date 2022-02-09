# Gradient of a scalar field

This example shows different methods to compute the gradient of a real-valued
3D scalar field $θ(\bm{x})$ in Fourier space, where $\bm{x} = (x, y, z)$.
It is assumed that the field is periodic with period $L = 2π$ along all
dimensions.

A working implementation of this example can be found in
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl).

## General procedure

The discrete Fourier expansion of $θ$ writes
```math
θ(\bm{x}) = ∑_{\bm{k} ∈ \Z^3} \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
```
where $\bm{k} = (k_x, k_y, k_z)$ are the Fourier wave numbers and $\hat{θ}$ is
the discrete Fourier transform of $θ$.
Then, the spatial derivatives of $θ$ are given by
```math
\frac{∂ θ(\bm{x})}{∂ x_i} =
∑_{\bm{k} ∈ \Z^3} i k_i \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
```
where the underscript $i$ denotes one of the spatial components $x$, $y$ or
$z$.

In other words, to compute $\bm{∇} θ = (∂_x θ, ∂_y θ, ∂_z θ)$, one has to:
1. transform $θ$ to Fourier space to obtain $\hat{θ}$,
2. multiply $\hat{θ}$ by $i \bm{k}$,
3. transform the result back to physical space to obtain $\bm{∇} θ$.

## Preparation

In this section, we initialise a random real-valued scalar field $θ$ and compute
its FFT.
For more details see the [Tutorial](@ref).

```@example gradient
using MPI
using PencilFFTs
using Random

MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
dims = (64, 32, 64)

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# Automatically create decomposition configuration
comm = MPI.COMM_WORLD
pen = Pencil(dims, comm)

# Create plan
plan = PencilFFTPlan(pen, transform)
```

```@example gradient
# Allocate data and initialise field
θ = allocate_input(plan)
randn!(θ)

# Perform distributed FFT
θ_hat = plan * θ

# Finally, we initialise the output that will hold ∇θ in Fourier space.
# Noting that ∇θ is a vector field, we choose to store it as a tuple of
# 3 PencilArrays.
# These two are exactly equivalent:
# ∇θ_hat = ntuple(d -> similar(θ_hat), Val(3))
∇θ_hat = allocate_output(plan, Val(3))
nothing # hide
```

## Fourier wave numbers

In general, the Fourier wave numbers are of the form
$k_i = 0, ±\frac{2π}{L_i}, ±\frac{4π}{L_i}, ±\frac{6π}{L_i}, \ldots$,
where $L_i$ is the period along dimension $i$.
When a real-to-complex Fourier transform is applied, roughly half of
these wave numbers are redundant due to the Hermitian symmetry of the complex
Fourier coefficients.
In practice, this means that for the fastest dimension $x$ (along which
a real-to-complex transform is performed), the negative wave numbers are
dropped, i.e. $k_x = 0, \frac{2π}{L_x}, \frac{4π}{L_x}, \ldots$.

The `AbstractFFTs` package provides a convenient way to generate the Fourier
wave numbers, using the functions
[`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq)
and
[`rfftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfftfreq).
We can use these functions to initialise a "grid" of wave numbers associated to
our 3D real-to-complex transform:
```@example gradient
using AbstractFFTs: fftfreq, rfftfreq

box_size = (2π, 2π, 2π)  # Lx, Ly, Lz
sample_rate = 2π .* dims ./ box_size

# In our case (Lx = 2π and Nx even), this gives kx = [0, 1, 2, ..., Nx/2].
kx = rfftfreq(dims[1], sample_rate[1])

# In our case (Ly = 2π and Ny even), this gives
# ky = [0, 1, 2, ..., Ny/2-1, -Ny/2, -Ny/2+1, ..., -1] (and similarly for kz).
ky = fftfreq(dims[2], sample_rate[2])
kz = fftfreq(dims[3], sample_rate[3])

kvec = (kx, ky, kz)
nothing # hide
```
Note that `kvec` now contains the wave numbers associated to the global domain.
In the following, we will only need the wave numbers associated to the portion
of the domain handled by the local MPI process.

## [Method 1: global views](@id gradient_method_global)

[`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray)s, returned for instance by [`allocate_input`](@ref)
and  [`allocate_output`](@ref), take indices that start at 1, regardless of the
location of the subdomain associated to the local process on the global grid.
(We say that `PencilArray`s take *local* indices.)
On the other hand, we have defined the wave number vector `kvec` which,
for each MPI process, is defined over the global domain, and as such it takes
*global* indices.

One straightforward way of making data arrays compatible with wave numbers is
to use global views, i.e. arrays that take global indices.
These are generated from `PencilArray`s by calling the [`global_view`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.global_view-Tuple{PencilArray})
function.
Note that, in general, global indices do *not* start at 1 for a given MPI
process.
A given process will own a range of data given by indices in `(i1:i2, j1:j2,
k1:k2)`.

Once we have global views, we can combine data and wave numbers using the
portion of global indices owned by the local MPI process, as shown below.

```@example gradient
# Generate global views of PencilArrays.
θ_glob = global_view(θ_hat)
∇θ_glob = map(global_view, ∇θ_hat)  # we map over the 3 elements of ∇θ_hat

# We can use CartesianIndices to iterate over the global indices associated to
# the local process.
for I in CartesianIndices(θ_glob)
    i, j, k = Tuple(I)  # unpack indices

    # Wave number vector associated to current Cartesian index.
    local kx, ky, kz  # hide
    kx = kvec[1][i]
    ky = kvec[2][j]
    kz = kvec[3][k]

    # Compute gradient in Fourier space.
    # Note that modifying ∇θ_glob also modifies the original PencilArray ∇θ_hat.
    ∇θ_glob[1][I] = im * kx * θ_glob[I]
    ∇θ_glob[2][I] = im * ky * θ_glob[I]
    ∇θ_glob[3][I] = im * kz * θ_glob[I]
end
```

The above loop can be written in a more efficient manner by precomputing
`im * θ_glob[I]` and by avoiding indexation with the `CartesianIndex` `I`,
using linear indexing instead:[^1]
```@example gradient
@inbounds for (n, I) in enumerate(CartesianIndices(θ_glob))
    i, j, k = Tuple(I)

    local kx, ky, kz  # hide
    kx = kvec[1][i]
    ky = kvec[2][j]
    kz = kvec[3][k]

    u = im * θ_glob[n]

    ∇θ_glob[1][n] = kx * u
    ∇θ_glob[2][n] = ky * u
    ∇θ_glob[3][n] = kz * u
end
```
This is basically the implementation of `gradient_global_view!` in
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl).
Also note that the above can be easily written in a more generic way, e.g. for
arbitrary dimensions, thanks in part to the use of `CartesianIndices`.
Moreover, in the above there is no notion of the dimension permutations
discussed in [the tutorial](@ref tutorial:output_data_layout), as it is all
hidden behind the implementation of `PencilArray`s.

Finally, we can perform a backwards transform to obtain $\bm{∇} θ$ in physical
space:
```@example gradient
∇θ = plan \ ∇θ_hat
nothing # hide
```
Note that the transform is automatically broadcast over the three fields
of the `∇θ_hat` vector, and the result `∇θ` is also a tuple of
three `PencilArray`s.

## [Method 2: explicit global indexing](@id gradient_method_global_explicit)

Sometimes, one does not need to write generic code.
In our case, one often knows the dimensionality of the problem and the
memory layout of the data (i.e. the underlying index permutation).

Below is a reimplementation of the above loop, using explicit indices instead
of `CartesianIndices`, and assuming that the underlying index permutation is
`(3, 2, 1)`, that is, data is stored in $(z, y, x)$ order.
As discussed in [the tutorial](@ref tutorial:output_data_layout),
this is the default for transformed arrays.
This example also serves as a clearer explanation for what is going on in the
[first method](@ref gradient_method_global).

```@example gradient
# Get local data range in the global grid.
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

    # Note that we still access the arrays in (i, j, k) order.
    # (The permutation happens behind the scenes!)
    u = im * θ_glob[i, j, k]

    ∇θ_glob[1][i, j, k] = kx * u
    ∇θ_glob[2][i, j, k] = ky * u
    ∇θ_glob[3][i, j, k] = kz * u
end
```

This implementation corresponds to `gradient_global_view_explicit!` in
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl).
Perhaps surprisingly, this implementation of the gradient is the fastest of all
tested.
(See the [benchmark results](@ref gradient_benchmarks):
the ["local index" implementation below](@ref gradient_method_local) is about
20% slower, while [Method 1](@ref gradient_method_global) is 40% slower.)
Note that we don't even need to switch to linear indexing to obtain optimal
performance!
Apparently there's a lot of compiler optimisations going on specifically for
this function.
This is evident when running `julia` with the `-O1` optimisation level (the
default is `-O2`), in which case this implementation becomes much slower than
the others (tested with Julia 1.4.1).

## [Method 3: using local indices](@id gradient_method_local)

Alternatively, we can avoid global views and work directly on `PencilArray`s
using local indices that start at 1.
In this case, part of the strategy is to construct a "local" grid of wave
numbers that can also be accessed with local indices.
Moreover, to obtain the local data range associated to a `PencilArray`,
we call the [`range_local`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.Pencils.range_local-Tuple{Union{PencilArray,%20Union{Tuple{Vararg{A,N}%20where%20N},%20AbstractArray{A,N}%20where%20N}%20where%20A%3C:PencilArray}}) function.
Apart from these details, this method is very similar to the [first one](@ref
gradient_method_global).

```@example gradient
# Get local data range in the global grid.
rng = range_local(θ_hat)  # = (i1:i2, j1:j2, k1:k2)

# Local wave numbers: (kx[i1:i2], ky[j1:j2], kz[k1:k2]).
kvec_local = getindex.(kvec, rng)

@inbounds for (n, I) in enumerate(CartesianIndices(θ_hat))
    i, j, k = Tuple(I)  # local indices

    # Wave number vector associated to current Cartesian index.
    local kx, ky, kz
    kx = kvec_local[1][i]
    ky = kvec_local[2][j]
    kz = kvec_local[3][k]

    u = im * θ_hat[n]

    ∇θ_hat[1][n] = kx * u
    ∇θ_hat[2][n] = ky * u
    ∇θ_hat[3][n] = kz * u
end
```

This implementation corresponds to `gradient_local!` in
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl).
Like [Method 1](@ref gradient_method_global), this implementation uses
`CartesianIndices` and can be made more generic with little effort.
In particular, there is no explicit use of index permutations, and no
assumptions need to be made in that regard.
In our tests, this implementation is about 15% faster than [Method 1](@ref
gradient_method_global), while still being generic and almost equally simple.

In
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl),
additional implementations using local indices can be found which rely on
a more advanced understanding of permutations and on the internals of the
[`PencilArrays`](https://jipolanco.github.io/PencilArrays.jl/dev/) package.
See for instance `gradient_local_parent!`, which directly works with the raw
data stored in Julia `Array`s; or `gradient_local_linear!`, which completely
avoids `CartesianIndices` while staying generic and efficient. We have found
that these display roughly the same performance as the example
above.

## Summary

The `PencilArrays` module provides different alternatives to deal with
MPI-distributed data that may be subject to dimension permutations.
In particular, one can choose to work with *global* indices (first two
examples) or with *local* indices (third example).

If one wants to stay generic, making sure that the same code will work for
arbitrary dimensions and will be efficient regardless of the underlying
dimension permutation, methods [1](@ref gradient_method_global) and [3](@ref
gradient_method_local) should be preferred.
These use `CartesianIndices` and make no assumptions on the permutations
(actually, permutations are completely invisible in the implementations).
[Method 3](@ref gradient_method_local) is faster and should be preferred for
performance.

The [second method](@ref gradient_method_global_explicit) uses explicit
`(i, j, k)` indices.
It assumes that the underlying permutation is `(3, 2, 1)` to loop with `i` as
the *slowest* index and `k` as the *fastest*, which is the optimal order in
this case given the permutation.
As such, the implementation is less generic than the others, but is slightly
easier to read.

The second method achieves better performance than the other implementations
(about 20% faster than [Method 3](@ref gradient_method_local)).
The difference between methods 2 and 3 is explained by a more efficient access
to the wave numbers `(kx, ky, kz)` in the former, since wave numbers are
contained in lazy `Frequencies` objects (returned by `rfftfreq` and `fftfreq`),
while in the local method, the wave numbers are collected into `Vector`s.
See the next section for more details.
Note that for larger problem sizes, the performance differences
between methods become negligible.

## [Benchmark results](@id gradient_benchmarks)

The following are the benchmark results obtained from running
[`examples/gradient.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/gradient.jl)
on a laptop, using 2 MPI processes and Julia 1.4.1, with an input array of
global dimensions ``64 × 32 × 64``.
The three methods detailed above are marked on the right.
The "lazy" marks indicate runs where the wave numbers were represented by
lazy `Frequencies` objects. Otherwise, they were collected into `Vector`s.

        gradient_global_view!...                  184.696 μs
        gradient_global_view! (lazy)...           169.519 μs  [Method 1]
        gradient_global_view_explicit!...         147.647 μs
        gradient_global_view_explicit! (lazy)...  122.157 μs  [Method 2]
        gradient_local!...                        146.937 μs  [Method 3]
        gradient_local_parent!...                 146.696 μs
        gradient_local_linear!...                 146.565 μs
        gradient_local_linear_explicit!...        147.078 μs


[^1]:
    This assumes that `CartesianIndices(θ_glob)` iterates in the order of the
    array elements in memory. This is not trivial when the array dimensions are
    permuted (which is the default for transformed arrays in PencilFFTs), and
    it actually wasn't the case until PencilFFTs `v0.2.0`.
