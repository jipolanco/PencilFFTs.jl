module PencilFFTs

include("Pencils/Pencils.jl")

using .Pencils

import FFTW
import MPI

# Type definitions
const FFTReal = FFTW.fftwReal  # Union{Float64,Float32}

end # module
