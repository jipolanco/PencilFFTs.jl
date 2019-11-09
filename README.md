# PencilFFTs.jl

[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://pages.oca.eu/jpolanco/pencilffts.jl/public)

Fast Fourier transforms of MPI-distributed Julia arrays.

This package provides functionality to distribute multidimensional arrays among
MPI processes, and to perform multidimensional FFTs (and related transforms) on
them.
This functionality is typically required for numerically solving physical
equations using pseudo-spectral methods in high-performance computing clusters
(a classical example is the Navier-Stokes equations describing hydrodynamic
turbulence).

The name of this package originates from the decomposition of 3D domains along
two out of three dimensions, sometimes called *pencil* decomposition.
This is illustrated by the figure below
([source](https://hal.archives-ouvertes.fr/tel-02084215v1)).
Typically, one wants to compute FFTs on a scalar or vector field along the
three spatial dimensions.
In the case of a pencil decomposition, 3D FFTs are performed one dimension at
a time (along the non-decomposed direction, using a serial FFT implementation).
Global data transpositions are then needed to switch from one pencil
configuration to the other and perform FFTs along the other dimensions.

![Pencil decomposition of 3D domains.](docs/src/img/pencils.svg)

The package is implemented in an efficient generic way that allows to decompose
any `N`-dimensional geometry along `M < N` dimensions (for the pencil
decomposition described above, `N = 3` and `M = 2`). Moreover, the transforms
applied along each dimension can be arbitrarily chosen among those supported by
`FFTW`, including complex-to-complex, real-to-complex, and (very soon)
real-to-real transforms.

## Similar projects

- [P3DFFT](https://www.p3dfft.net) implements parallel 3D FFTs using pencil
  decomposition in Fortran and C++.

- [2DECOMP&FFT](http://www.2decomp.org) is another parallel 3D FFT library
  using pencil decomposition written in Fortran.
