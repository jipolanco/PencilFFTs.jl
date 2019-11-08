"""
Pencil module for 2D decomposition of 3D domains using MPI.

Handles different "pencil" decomposition configurations and data transpositions
between them.
"""
module Pencils

using MPI
using TimerOutputs

import Base: ndims, size, length, eltype, show
import LinearAlgebra: transpose!

export Pencil, PencilArray, MPITopology
export gather
export get_comm, get_decomposition, get_permutation, get_timer
export ndims_extra
export size_local, size_global
export transpose!

# Describes the portion of an array held by a given MPI process.
# TODO maybe use CartesianIndices?
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# Describes indices in an array as a tuple.
const Indices{N} = NTuple{N,Int} where N

const Permutation{N} = NTuple{N,Int} where N
const OptionalPermutation{N} = Union{Nothing, Permutation{N}} where N

# Type definitions
include("mpi_topology.jl")  # MPITopology
include("pencil.jl")        # Pencil
include("arrays.jl")        # PencilArray

include("data_ranges.jl")
include("mpi_wrappers.jl")
include("permutations.jl")
include("transpose.jl")

end
