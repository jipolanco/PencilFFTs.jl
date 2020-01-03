# TODO

- Break into multiple packages: PencilArrays, PencilFFTs (maybe also
  PencilIterators / PencilGrids? PencilIO?)

- `extra_dims` stuff should be in a separate section of the docs (or
  removed...)

- Avoid transpositions when using `NoTransform`

- Is it possible to reduce array allocations (buffers)?

- Disallow (or support?) initialising BRFFT plan. Instead, RFFT plan should be
  created and applied backwards.

- Multithreading?

- Implement FFT padding (like in the Fortran P3DFFT), e.g. for 2/3 dealiasing.

- Add `Grids` module?

- Permuted dimensions and CartesianIndices: visiting order of array and of grid
  must be the same.

  * Performance: can I make CartesianIndices visit permuted arrays in memory
    order? (this is not currently the case...)

## Pencils

- add optional callbacks to `transpose!`? To do stuff while data is being received...

- functions to exchange ghost data between pencils

- parallel HDF5 I/O (optional?) -> use Requires
