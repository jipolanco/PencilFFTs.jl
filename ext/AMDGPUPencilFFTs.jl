module AMDGPUPencilFFTs

using .Transforms
using AMDGPU
if AMDGPU.functional(:rocfft)
    using AMDGPU: rocFFT, AnyROCArray
end

if AMDGPU.functional(:rocfft)
    # c2c.jl
    plan(::FFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft(A, args...; kwargs...)
    plan(::FFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft!(A, args...; kwargs...)
    plan(::BFFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft(A, args...; kwargs...)
    plan(::BFFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft!(A, args...; kwargs...)

    # r2c.jl
    plan(::RFFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_rfft(A, args...; kwargs...)

    function plan(tr::BRFFT, A::AnyROCArray, dims; kwargs...)
        Nin = size(A, first(dims))  # input length along first dimension
        d = length_output(tr, Nin)
        rocFFT.plan_brfft(A, d, dims; kwargs...)
    end

end

end # module