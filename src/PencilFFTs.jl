module PencilFFTs

import FFTW
import MPI

using LinearAlgebra
using Reexport
using TimerOutputs

@reexport using PencilArrays

include("Transforms/Transforms.jl")
@reexport using .Transforms

import PencilArrays.Transpositions: AbstractTransposeMethod
import .Transforms: AbstractTransform, FFTReal, scale_factor

export PencilFFTPlan
export allocate_input, allocate_output, scale_factor

# Deprecated in v0.10
@deprecate get_scale_factor scale_factor

# Functions to be extended for PencilFFTs types.
import PencilArrays: get_comm, timer

const AbstractTransformList{N} = NTuple{N, AbstractTransform} where N

include("global_params.jl")
include("plans.jl")
include("allocate.jl")
include("operations.jl")

end # module
