# TODO

- Reduce array allocations (buffers)

- Vector (or tensor) transforms
  * use fastest or slowest indices for components?

- Add benchmarks: compare with P3DFFT, ...

- Disallow (or support?) initialising BRFFT plan. Instead, RFFT plan should be
  created and applied backwards.

## Docs

- How should I access the data? index permutations?

## For later

- Compatibility with [MKL FFTW3 wrappers](https://software.intel.com/en-us/mkl-developer-reference-c-using-fftw3-wrappers)?
  See also [here](https://github.com/JuliaMath/FFTW.jl#mkl).

- Multithreading?

## Pencils

- add optional callbacks to `transpose!`? To do stuff while data is being received...

- functions to exchange ghost data between pencils

- parallel HDF5 I/O (optional?)
