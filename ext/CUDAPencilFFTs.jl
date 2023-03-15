module CUDAPencilFFTs

using .Transforms
using CUDA

# c2c.jl
@show 212341

plan(::FFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft(A, args...; kwargs...)
plan(::FFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_fft!(A, args...; kwargs...)
plan(::BFFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft(A, args...; kwargs...)
plan(::BFFT!, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_bfft!(A, args...; kwargs...)

# r2c.jl

plan(::RFFT, A::AnyCuArray, args...; kwargs...) = CUFFT.plan_rfft(A, args...; kwargs...)

function plan(tr::BRFFT, A::AnyCuArray, dims; kwargs...)
    Nin = size(A, first(dims))  # input length along first dimension
    d = length_output(tr, Nin)
    CUFFT.plan_brfft(A, d, dims; kwargs...)
end

end # module