#!/usr/bin/env julia

# In-place transforms.

using FFTW
using MPI
using PencilFFTs

using Random: randn!

const INPUT_DIMS = (64, 32, 64)

MPI.Init()

dims = INPUT_DIMS

# Combine r2r and c2c in-place transforms.
transforms = (
    Transforms.R2R!(FFTW.REDFT01),
    Transforms.FFT!(),
    Transforms.R2R!(FFTW.DHT),
)

# MPI topology information
comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# Let's do a 1D decomposition.
proc_dims = (Nproc, )

# Create in-place plan
plan = PencilFFTPlan(dims, transforms, proc_dims, comm)
rank == 0 && println(plan)
@assert Transforms.is_inplace(plan)

# Allocate data for the plan.
# This returns a `ManyPencilArray` container that holds multiple
# `PencilArray` views.
A = allocate_input(plan) :: PencilArrays.ManyPencilArray

# The input output `PencilArray`s are recovered using `first` and
# `last`.
u_in = first(A) :: PencilArray
u_out = last(A) :: PencilArray

# Initialise input data.
randn!(u_in)

# Apply in-place forward transform on the `ManyPencilArray` container.
plan * A

# After the transform, operations should be performed on the output view
# `u_out`. For instance, let's compute the global sum of the transformed data.
sum_local = sum(u_out)
sum_global = MPI.Allreduce(sum_local, +, comm)

# Apply in-place backward transform.
plan \ A

# Now we can again perform operations on the input view `u_in`...
