module AMDGPUPencilFFTs

using PencilFFTs
using PencilFFTs.Transforms: FFT, FFT!, BFFT, BFFT!, RFFT, BRFFT
using AMDGPU
if AMDGPU.functional(:rocfft)
    using AMDGPU: rocFFT, AnyROCArray
end

if AMDGPU.functional(:rocfft)
    # c2c.jl
    PencilFFTs.Transforms.plan(::FFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft(A, args...; kwargs...)
    PencilFFTs.Transforms.plan(::FFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_fft!(A, args...; kwargs...)
    PencilFFTs.Transforms.plan(::BFFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft(A, args...; kwargs...)
    PencilFFTs.Transforms.plan(::BFFT!, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_bfft!(A, args...; kwargs...)

    # r2c.jl
    PencilFFTs.Transforms.plan(::RFFT, A::AnyROCArray, args...; kwargs...) = rocFFT.plan_rfft(A, args...; kwargs...)

    function PencilFFTs.Transforms.plan(tr::BRFFT, A::AnyROCArray, dims; kwargs...)
        Nin = size(A, first(dims))  # input length along first dimension
        d = length_output(tr, Nin)
        rocFFT.plan_brfft(A, d, dims; kwargs...)
    end

end

end # module