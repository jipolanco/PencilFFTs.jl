# TODO

## FFTs

- Include FFTW plans in PencilPlan

- Reduce array allocations (buffers)

- Vector (or tensor) transforms
  * use fastest or slowest indices for components?

- Define `*` and `mul!` (and also `assert_applicable`) like in `FFTW.jl`

## For later

- Generalise transform types: add `c2c`, `r2r`, ...

- Compatibility with [MKL FFTW3 wrappers](https://software.intel.com/en-us/mkl-developer-reference-c-using-fftw3-wrappers)?
  See also [here](https://github.com/JuliaMath/FFTW.jl#mkl).

- Multithreading?

## Pencils

- add optional callbacks to `transpose!`. This could be used to compute FFTs
  while data is being received.

- functions to exchange ghost data between pencils

- parallel HDF5 I/O (optional?)

- try to add new `MPI.Cart_*` functions to `MPI.jl`

- compatibility with other distributed array packages? (`MPIArrays`,
  `DistributedArrays`?)

## Other ideas

- Define arrays with [custom
  ranges](https://docs.julialang.org/en/v1.2/devdocs/offset-arrays), so that they take global indices.
