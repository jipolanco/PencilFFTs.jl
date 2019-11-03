#!/usr/bin/env julia

using PencilFFTs
import PencilFFTs: PencilArray

using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test

function test_transform_types(size_in)
    transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
    fft_params = PencilFFTs.GlobalFFTParams(size_in, transforms)

    @test fft_params isa PencilFFTs.GlobalFFTParams{Float64, 3,
                                                    typeof(transforms)}
    @test inv(Transforms.RFFT()) === Transforms.BRFFT()
    @test inv(Transforms.IRFFT()) === Transforms.RFFT()

    transforms_inv = inv.(transforms)
    size_out = Transforms.length_output.(transforms, size_in)

    @test transforms_inv ===
        (Transforms.BRFFT(), Transforms.BFFT(), Transforms.BFFT())
    @test size_out === (size_in[1] ÷ 2 + 1, size_in[2:end]...)
    @test Transforms.length_output.(transforms_inv, size_out) === size_in

    @test PencilFFTs.input_data_type(fft_params) === Float64

    nothing
end

function test_pencil_plans(size_in::Tuple)
    @assert length(size_in) >= 3
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    # Let MPI_Dims_create choose the decomposition.
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
    plan = PencilFFTPlan(size_in, transforms, proc_dims, comm, Float64)

    # This interface will change...
    let u = PencilArray(first(plan.plans).pencil_in)
        randn!(u)
        v = PencilArray(last(plan.plans).pencil_out)
        mul!(v, plan, u)
        @time mul!(v, plan, u)
    end

    if Nproc == 1
        # @code_warntype PencilFFTPlan(size_in, transforms, proc_dims, comm)
        # @code_warntype PencilFFTs._create_pencils(plan.global_params,
        #                                           plan.topology)
        # @code_warntype PencilFFTs.input_data_type(Float64, transforms...)

        let u = PencilArray(first(plan.plans).pencil_in)
            @code_warntype PencilFFTs._apply_plans(u, plan.plans...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.FFT())
            @test PencilFFTs.input_data_type(Float32, transforms...) ===
                ComplexF32
            # @code_warntype PencilFFTs.input_data_type(Float32, transforms...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.NoTransform())
            @test PencilFFTs.input_data_type(Float32, transforms...) ===
                Nothing
            # @code_warntype PencilFFTs.input_data_type(Float32, transforms...)
        end
    end

    nothing
end

function main()
    MPI.Init()

    size_in = (16, 24, 32)
    test_transform_types(size_in)
    test_pencil_plans(size_in)

    MPI.Finalize()
end

main()
