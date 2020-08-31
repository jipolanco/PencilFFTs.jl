module PencilFFTs

import FFTW
import MPI

using LinearAlgebra
using Reexport
using TimerOutputs

@reexport using PencilArrays

include("Transforms/Transforms.jl")
@reexport using .Transforms

# For convenience...
import PencilArrays.Transpositions: AbstractTransposeMethod
import .Transforms: AbstractTransform, FFTReal

export PencilFFTPlan
export allocate_input, allocate_output, get_scale_factor

# Functions to be extended for PencilFFTs types.
import PencilArrays: get_comm, get_timer

const AbstractTransformList{N} = NTuple{N, AbstractTransform} where N

include("global_fft.jl")
include("pencil_plans.jl")
include("operations.jl")

end # module
