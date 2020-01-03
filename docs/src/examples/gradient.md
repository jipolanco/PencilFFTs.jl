# Gradient of a scalar field

This example shows different methods to compute the gradient of a real-valued
3D scalar field $θ(\bm{x})$ in Fourier space, where $\bm{x} = (x, y, z)$.
It is assumed that the field is periodic with period $L = 2π$ along all
dimensions.

## General procedure

The discrete Fourier expansion of $θ$ writes
```math
θ(\bm{x}) = ∑_{\bm{k} ∈ \Z} \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
```
where $\bm{k} = (k_x, k_y, k_z)$ are the Fourier wave numbers and $\hat{θ}$ is
the discrete Fourier transform of $θ$.
Then, the spatial derivatives of $θ$ are given by
```math
\frac{∂ θ(\bm{x})}{∂ x_i} =
∑_{\bm{k} ∈ \Z} i k_i \hat{θ}(\bm{k}) \, e^{i \bm{k} ⋅ \bm{x}},
```
where the underscript $i$ denotes one of the spatial components $x$, $y$ or
$z$.

In other words, to compute $∇θ = (∂_x θ, ∂_y θ, ∂_z θ)$, one has to:
1. transform $θ$ to Fourier space to obtain $\hat{θ}$,
2. multiply $\hat{θ}$ by $i \bm{k}$,
3. transform the result back to physical space to obtain $∇θ$.

## Preparation

In this section, we initialise a random real-valued scalar field $θ$ and compute
its FFT.
For more details see the [Tutorial](@ref).

```julia
using MPI
using PencilFFTs
using Random

MPI.Init()

# Input data dimensions (Nx × Ny × Nz)
dims = (16, 32, 64)

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.RFFT()

# MPI topology information
comm = MPI.COMM_WORLD  # we assume MPI.Comm_size(comm) == 12
proc_dims = (3, 4)     # 3 processes along `y`, 4 along `z`

# Create plan
plan = PencilFFTPlan(dims, transform, proc_dims, comm)

# Allocate data and initialise field
θ = allocate_input(plan)
randn!(θ)

# Perform distributed FFT
θ_hat = plan * θ
```

## Fourier wave numbers

In general, the Fourier wave numbers are of the form
$k_i = 0, ±\frac{2π}{L_i}, ±\frac{4π}{L_i}, ±\frac{6π}{L_i}, \ldots$,
where $L_i$ is the period along dimension $i$.
However, when a real-to-complex Fourier transform is applied, roughly half of
these wave numbers are redundant due to the Hermitian symmetry of the complex
Fourier coefficients.
In practice, this means that for the fastest dimension $x$ (along which
a real-to-complex transform is performed), only the positive wave numbers are
kept, i.e. $k_x = 0, \frac{2π}{L_x}, \frac{4π}{L_x}, \ldots$.

The `AbstractFFTs` package provides a convenient way to generate the Fourier
wave numbers, using the functions
[`fftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.fftfreq)
and
[`rfftfreq`](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.rfftfreq).
We can use these functions to initialise a "grid" of wave numbers associated to
our 3D real-to-complex transform:
```julia
using AbstractFFTs: fftfreq, rfftfreq

box_size = (2π, 2π, 2π)  # L_x, L_y, L_z
sample_rate = 2π .* dims ./ box_size

# In our case (L_x = 2π and N_x even), this gives kx = [0, 1, 2, ..., N_x/2].
kx = rfftfreq(dims[1], sample_rate[1])

# In our case (L_y = 2π and N_y even), this gives
# ky = [0, 1, 2, ..., N_y/2-1, -Ny/2, -Ny/2+1, ..., -1] (and similarly for kz).
ky = fftfreq(dims[2], sample_rate[2])
kz = fftfreq(dims[3], sample_rate[3])

kvec = (kx, ky, kz)
```
Note that `kvec` now contains the wave numbers associated to the global domain.
In the following, we will only need the wave numbers associated to the portion
of the domain handled by the local MPI process.

## Method 1: global views

Perhaps the most straightforward approach is to use global views, i.e. arrays
that take global indices.
