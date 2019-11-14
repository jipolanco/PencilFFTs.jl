module PencilFFTs

import FFTW
import MPI

using Reexport
using TimerOutputs

include("Pencils/Pencils.jl")
include("Transforms/Transforms.jl")

@reexport using .Pencils
@reexport using .Transforms

# For convenience...
import .Transforms: AbstractTransform, FFTReal
import .TransposeMethods: AbstractTransposeMethod

export PencilFFTPlan
export allocate_input, allocate_output

# Functions to be extended for PencilFFTs types.
import .Pencils: get_comm, get_timer

import Base: *, \, show
import LinearAlgebra: mul!, ldiv!

const AbstractTransformList{N} = NTuple{N, AbstractTransform} where N

include("global_fft.jl")
include("pencil_plans.jl")
include("operations.jl")

end # module
