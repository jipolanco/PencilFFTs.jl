# Global MPI operations

```@meta
CurrentModule = PencilFFTs.PencilArrays
```

One of the most time-consuming parts of a large-scale computation involving
multidimensional FFTs, is the global data transpositions between different MPI
decomposition configurations.
In `PencilArrays`, this is performed by the
[`transpose!`](@ref Transpositions.transpose!) function, which
takes two `PencilArray`s, typically associated to two different configurations.
The implementation performs comparably to similar implementations in
lower-level languages (see [Benchmarks](@ref)).

Also provided is a [`gather`](@ref) function that creates a single global array
from decomposed data.
This can be useful for tests (in fact, it is used in the `PencilArrays` tests to
verify the correctness of the transpositions), but shouldn't be used with large
datasets.
It is generally useful for small problems where the global size of the data can
easily fit the locally available memory.

```@docs
Transpositions.Transposition
Transpositions.transpose!
MPI.Waitall!
gather
```
