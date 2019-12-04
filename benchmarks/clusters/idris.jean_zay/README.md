Tests run on the [Jean-Zay cluster](http://www.idris.fr/jean-zay/jean-zay-presentation.html)
([in English](http://www.idris.fr/eng/jean-zay/cpu/jean-zay-cpu-hw-eng.html)) of
IDRIS (France).

Some relevant specifications:

- HPE SGI 8600 computer
- 1528 XA730i compute nodes, with 2 Intel Cascade Lake 6248 processors (20
  cores at 2.5 GHz), or 61120 cores available

The benchmarks were run using Intel MPI 2019.0.4 and FFTW 3.3.9 (version
bundled by FFTW.jl).

Date: 4 December 2019 (PencilFFTs commit
[`3f238172a`](https://github.com/jipolanco/PencilFFTs.jl/commit/3f238172a62036104f535d7bb22933096458f9a8)).
