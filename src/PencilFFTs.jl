module PencilFFTs

include("Pencils/Pencils.jl")
include("Transforms/Transforms.jl")

using .Pencils
using .Transforms

# For convenience...
import .Transforms: AbstractTransform

export PencilFFTPlan
export Transforms

import FFTW
import MPI

const FFTReal = FFTW.fftwReal  # = Union{Float32, Float64}

include("global_fft.jl")
include("pencil_plans.jl")

end # module
