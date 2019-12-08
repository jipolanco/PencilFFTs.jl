"""
Module for multidimensional data decomposition using MPI.

Handles different decomposition configurations and data transpositions between
them. Also defines relevant data structures for handling distributed data.
"""
module Pencils

using MPI
using OffsetArrays
using Reexport
using TimerOutputs

import Base: @propagate_inbounds
import LinearAlgebra

export Pencil, PencilArray, MPITopology
export PencilArrayCollection
export pencil
export gather
export get_comm, get_decomposition, get_permutation, get_timer
export global_view
export ndims_extra, ndims_space, extra_dims
export range_local, size_local, size_global, spatial_indices
export transpose!

# Describes the portion of an array held by a given MPI process.
const ArrayRegion{N} = NTuple{N,UnitRange{Int}} where N

# Modules
include("MPITopologies.jl")
using .MPITopologies
import .MPITopologies: get_comm

# Type definitions
include("pencil.jl")       # Pencil
include("arrays.jl")       # PencilArray
include("global_view.jl")  # GlobalPencilArray

include("data_ranges.jl")
include("mpi_wrappers.jl")
include("permutations.jl")
include("transpose.jl")

end
