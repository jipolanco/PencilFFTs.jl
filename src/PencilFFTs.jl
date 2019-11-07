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

export PencilFFTPlan
export allocate_input, allocate_output

# Functions to be extended for PencilFFTs types.
import .Pencils: get_comm, get_timer

# Operators for applying direct and inverse plans (same as in AbstractFFTs).
import Base: *, \
import LinearAlgebra: mul!, ldiv!

include("global_fft.jl")
include("pencil_plans.jl")
include("operations.jl")

end # module
