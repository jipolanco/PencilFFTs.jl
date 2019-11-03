module PencilFFTs

include("Pencils/Pencils.jl")
include("Transforms/Transforms.jl")

using .Pencils
using .Transforms

# For convenience...
import .Transforms: AbstractTransform, FFTReal

export PencilFFTPlan
export Transforms

import FFTW
import MPI

# Operators for applying direct and inverse plans (same as in AbstractFFTs).
import Base: *, \
import LinearAlgebra: mul!, ldiv!

include("global_fft.jl")
include("pencil_plans.jl")
include("operations.jl")

end # module
