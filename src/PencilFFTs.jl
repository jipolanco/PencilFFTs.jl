module PencilFFTs

include("Pencils/Pencils.jl")
include("Transforms/Transforms.jl")

using .Pencils
using .Transforms

export Transforms

import FFTW
import MPI

# Type definitions
const FFTReal = FFTW.fftwReal  # Union{Float64,Float32}

include("global_fft.jl")

end # module
