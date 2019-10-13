# TODO

- Include FFTW plans in PencilPlan

- Reduce array allocations (buffers)

- Split `PencilPlan`: define `PlanData` and `FFTPlans` types.
  Or define a type for each pencil configuration / 1D transform.

- Vector (or tensor) transforms
  * use fastest or slowest indices for components?

- Define `*` and `mul!` (and also `assert_applicable`) like in `FFTW.jl`

## For later

- Generalise transform types: add `c2c`, `r2r`, ...

- Compatibility with [MKL FFTW3 wrappers](https://software.intel.com/en-us/mkl-developer-reference-c-using-fftw3-wrappers)?
  See also [here](https://github.com/JuliaMath/FFTW.jl#mkl).

- Multithreading?

- Special treatment for slab (1D) decomposition

## Pencils

- functions to exchange ghost data between pencils

- parallel HDF5 I/O (optional?)

- function to gather values from all processes. May be useful for testing...

- try to add new `MPI.Cart_*` functions to `MPI.jl`

## Other ideas

- Define arrays with [custom
  ranges](https://docs.julialang.org/en/v1.2/devdocs/offset-arrays), so that they take global indices.
