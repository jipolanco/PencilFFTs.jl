# In-place transforms

Complex-to-complex and real-to-real transforms can be performed in-place,
enabling important memory savings.
The procedure is very similar to that of out-of-place transforms described in
[the tutorial](@ref Tutorial).
The differences are illustrated in the sections below.

A working implementation of this example can be found in
[`examples/in-place.jl`](https://github.com/jipolanco/PencilFFTs.jl/blob/master/examples/in-place.jl).

## Creating plans

To create an in-place plan, pass an in-place transform such as
[`Transforms.FFT!`](@ref) or [`Transforms.R2R!`](@ref) to
[`PencilFFTPlan`](@ref).
For instance:

```julia
dims = (16, 32, 64)

# Apply a 3D in-place complex-to-complex FFT.
transform = Transforms.FFT!()

# One can also combine different types of in-place transforms.
# For instance:
# transform = (
#     Transforms.R2R!{FFTW.REDFT01}(),
#     Transforms.FFT!(),
#     Transforms.R2R!{FFTW.DHT}(),
# )

comm = MPI.COMM_WORLD
Nproc = MPI.Comm_size(comm)
proc_dims = (Nproc, )  # let's perform a 1D decomposition

# Create plan
plan = PencilFFTPlan(dims, transform, proc_dims, comm)
```

Note that in-place real-to-complex transforms are not currently supported.
(In other words, the `RFFT!` transform type is not defined.)

## Allocating data

As with out-of-place plans, data should be allocated using
[`allocate_input`](@ref).
The difference is that, for in-place plans, this function returns
a [`ManyPencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.ManyPencilArray) object, which is a container holding multiple
[`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) views sharing the same memory space.

```julia
# Allocate data for the plan.
# Since `plan` is in-place, this returns a `ManyPencilArray` container.
A = allocate_input(plan)
```

Note that [`allocate_output`](@ref) also works for in-place plans, but it
returns exactly the same thing as `allocate_input`.

As shown in the next section, in-place plans must be applied on the returned
`ManyPencilArray`.
On the other hand, one usually wants to access and modify data, and for this
one needs the `PencilArray` views contained in the `ManyPencilArray`.
The input and output array views can be obtained by calling
[`first(::ManyPencilArray)`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#Base.first-Tuple{ManyPencilArray}) and [`last(::ManyPencilArray)`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#Base.last-Tuple{ManyPencilArray}).

For instance, we can initialise the input array with some data before
transforming:
```julia
using Random
u_in = first(A)  # input data view
randn!(u_in)
```

## Applying plans

Like in `FFTW.jl`, one can perform in-place transforms using the `*` and
`\ ` operators.
As mentioned above, in-place plans must be applied on the `ManyPencilArray`
containers returned by `allocate_input`.
```julia
plan * A  # perform in-place forward transform
```

After performing an in-place transform, we usually want to do operations on the
output data.
For instance, let's compute the global sum of the transformed data:
```julia
u_out = last(A)         # output data view
sum_local = sum(u_out)  # sum of transformed data on local MPI process
sum_global = MPI.Allreduce(sum_local, +, comm)
```

Finally, we can perform a backward transform and do stuff with the input view:
```julia
plan \ A  # perform in-place backward transform

# Now we can again do stuff with the input view `u_in`...
```
