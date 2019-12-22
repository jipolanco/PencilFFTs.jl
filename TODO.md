# TODO

- Avoid transpositions when using `NoTransform`

- Reduce array allocations (buffers)

- Disallow (or support?) initialising BRFFT plan. Instead, RFFT plan should be
  created and applied backwards.

- Multithreading?

- Implement FFT padding (like in the Fortran P3DFFT), e.g. for 2/3 dealiasing.

- Add `Grids` module.

## Pencils

- add optional callbacks to `transpose!`? To do stuff while data is being received...

- functions to exchange ghost data between pencils

- parallel HDF5 I/O (optional?) -> use Requires
