#!/usr/bin/env julia

using PencilFFTs

using MPI

using Test

function main()
    MPI.Init()

    Nxyz = (16, 21, 41)
    transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
    fft_params = PencilFFTs.GlobalFFTParams(Nxyz, transforms)

    @test fft_params isa PencilFFTs.GlobalFFTParams{3, typeof(transforms)}
    @test inv(Transforms.RFFT()) === Transforms.BRFFT()
    @test inv(Transforms.IRFFT()) === Transforms.RFFT()
    @test inv.(transforms) ===
        (Transforms.BRFFT(), Transforms.BFFT(), Transforms.BFFT())

    MPI.Finalize()
end

main()
