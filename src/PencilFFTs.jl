module PencilFFTs

include("Pencils/Pencils.jl")
include("Transforms/Transforms.jl")

using .Pencils
using .Transforms

export PencilFFTPlan
export Transforms

import FFTW
import MPI

include("global_fft.jl")
include("pencil_plans.jl")

end # module
