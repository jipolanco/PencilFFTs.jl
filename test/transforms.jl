#!/usr/bin/env julia

using PencilFFTs

using MPI

using Test

function main()
    MPI.Init()

    size_in = (16, 21, 41)
    transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
    fft_params = PencilFFTs.GlobalFFTParams(size_in, transforms)

    @test fft_params isa PencilFFTs.GlobalFFTParams{3, typeof(transforms)}
    @test inv(Transforms.RFFT()) === Transforms.BRFFT()
    @test inv(Transforms.IRFFT()) === Transforms.RFFT()

    transforms_inv = inv.(transforms)
    size_out = Transforms.length_output.(transforms, size_in)

    @test transforms_inv ===
        (Transforms.BRFFT(), Transforms.BFFT(), Transforms.BFFT())
    @test size_out === (size_in[1] ÷ 2 + 1, size_in[2:end]...)
    @test Transforms.length_output.(transforms_inv, size_out) === size_in

    MPI.Finalize()
end

main()
