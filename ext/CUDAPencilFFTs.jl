module CUDAPencilFFTs

using PencilFFTs
using PencilFFTs.Transforms: FFT, FFT!, BFFT, BFFT!, RFFT, BRFFT
using CUDA

# c2c.jl
@show 212341

PencilFFTs.Transforms.plan(::FFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft(A, args...; kwargs...)
PencilFFTs.Transforms.plan(::FFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft!(A, args...; kwargs...)
PencilFFTs.Transforms.plan(::BFFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft(A, args...; kwargs...)
PencilFFTs.Transforms.plan(::BFFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft!(A, args...; kwargs...)

# r2c.jl

PencilFFTs.Transforms.plan(::RFFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_rfft(A, args...; kwargs...)

function PencilFFTs.Transforms.plan(tr::BRFFT, A::AnyCuArray, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    CUFFT.plan_brfft(A, d, dims; kwargs...)
end

end # module