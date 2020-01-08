#!/usr/bin/env julia

using PencilFFTs

import FFTW
using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test
using TimerOutputs

const DATA_DIMS = (64, 40, 32)

const DEV_NULL = @static Sys.iswindows() ? "nul" : "/dev/null"

const TEST_KINDS_R2R = Transforms.R2R_SUPPORTED_KINDS

function test_transform_types(size_in)
    transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
    fft_params = PencilFFTs.GlobalFFTParams(size_in, transforms)

    @testset "General transforms" begin
        @test fft_params isa PencilFFTs.GlobalFFTParams{Float64, 3,
                                                        typeof(transforms)}
        @test binv(Transforms.RFFT()) === Transforms.BRFFT()

        transforms_binv = binv.(transforms)
        size_out = Transforms.length_output.(transforms, size_in)

        @test transforms_binv ===
        (Transforms.BRFFT(), Transforms.BFFT(), Transforms.BFFT())
        @test size_out === (size_in[1] ÷ 2 + 1, size_in[2:end]...)
        @test Transforms.length_output.(transforms_binv, size_out) === size_in

        @test PencilFFTs.input_data_type(fft_params) === Float64
    end


    # Test type stability of generated plan_r2r (which, as defined in FFTW.jl,
    # is type unstable!). See comments of `plan` in src/Transforms/r2r.jl.
    @testset "r2r transforms" begin
        kind = FFTW.REDFT01
        transform = Transforms.R2R{kind}()
        transform! = Transforms.R2R!{kind}()

        let kind_inv = FFTW.REDFT10
            @test binv(transform) === Transforms.R2R{kind_inv}()
            @test binv(transform!) === Transforms.R2R!{kind_inv}()
        end

        A = zeros(4, 6, 8)
        for tr in (transform, transform!)
            @inferred Transforms.plan(tr, A, 2)
            @inferred Transforms.plan(tr, A, (1, 3))
            @inferred Transforms.plan(tr, A)

            # This will fail because length(2:3) is not known by the compiler.
            @test_throws ErrorException @inferred Transforms.plan(tr, A, 2:3)
        end

        for kind in (FFTW.R2HC, FFTW.HC2R), T in (Transforms.R2R, Transforms.R2R!)
            # Unsupported r2r kinds.
            @test_throws ArgumentError T{kind}()
            @test_throws ArgumentError T{kind}()
        end
    end

    nothing
end

function test_transforms(comm, proc_dims, size_in; extra_dims=())
    root = 0

    plan_kw = (:extra_dims => extra_dims, )
    N = length(size_in)

    make_plan(planner, args...; dims=1:N) = x -> planner(x, args..., dims)

    pair_r2r(tr::Transforms.R2R) =
        tr => make_plan(FFTW.plan_r2r, Transforms.kind(tr))
    pairs_r2r = (pair_r2r(Transforms.R2R{k}()) for k in TEST_KINDS_R2R)

    # TODO test c2c and some r2r in-place transforms

    pairs = (
             Transforms.FFT() => make_plan(FFTW.plan_fft),
             Transforms.RFFT() => make_plan(FFTW.plan_rfft),
             Transforms.BFFT() => make_plan(FFTW.plan_bfft),
             pairs_r2r...,
             (Transforms.NoTransform(), Transforms.RFFT(), Transforms.FFT())
                => make_plan(FFTW.plan_rfft, dims=2:3),
             (Transforms.FFT(), Transforms.NoTransform(), Transforms.FFT())
                => make_plan(FFTW.plan_fft, dims=(1, 3)),
             (Transforms.FFT(), Transforms.NoTransform(), Transforms.NoTransform())
                => make_plan(FFTW.plan_fft, dims=1),
             Transforms.BRFFT() => make_plan(FFTW.plan_brfft),  # not yet supported
            )

    @testset "$(p.first) -- $T" for p in pairs, T in (Float32, Float64)
        if p.first === Transforms.BRFFT()
            # FIXME...
            # In this case, I need to change the order of the transforms
            # (from right to left)
            @test_broken PencilFFTPlan(size_in, p.first, proc_dims, comm, T;
                                       plan_kw...)
            continue
        end

        @inferred PencilFFTPlan(size_in, p.first, proc_dims, comm, T;
                                plan_kw...)
        plan = PencilFFTPlan(size_in, p.first, proc_dims, comm, T;
                             plan_kw...)
        fftw_planner = p.second

        println("\n", "-"^60, "\n\n", plan, "\n")

        @inferred allocate_input(plan)
        @inferred allocate_input(plan, 2, 3)
        @inferred allocate_input(plan, Val(3))
        @inferred allocate_output(plan)
        u = allocate_input(plan)
        v = allocate_output(plan)

        randn!(u)

        mul!(v, plan, u)
        uprime = similar(u)
        ldiv!(uprime, plan, v)

        @test u ≈ uprime

        # Compare result with serial FFT.
        ug = gather(u, root)
        vg = gather(v, root)

        if ug !== nothing && vg !== nothing
            p = fftw_planner(ug)
            vg_serial = p * ug
            @test vg ≈ vg_serial
        end

        MPI.Barrier(comm)
    end

    nothing
end

function test_pencil_plans(size_in::Tuple, pdims::Tuple)
    @assert length(size_in) >= 3
    comm = MPI.COMM_WORLD
    myrank = MPI.Comm_rank(comm)
    myrank == 0 || redirect_stdout(open(DEV_NULL, "w"))

    @inferred PencilFFTPlan(size_in, Transforms.RFFT(), pdims, comm, Float64)

    @testset "Transform types" begin
        let transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
            @inferred PencilFFTPlan(size_in, transforms, pdims, comm)
            @inferred PencilFFTs.input_data_type(Float64, transforms...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.FFT())
            @test PencilFFTs.input_data_type(Float32, transforms...) ===
                ComplexF32
            @inferred PencilFFTs.input_data_type(Float32, transforms...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.NoTransform())
            @test PencilFFTs.input_data_type(Float32, transforms...) ===
                Nothing
            @inferred PencilFFTs.input_data_type(Float32, transforms...)
        end
    end

    test_transforms(comm, pdims, size_in, extra_dims=(3, ))
    test_transforms(comm, pdims, size_in)

    redirect_stdout(stdout)  # undo redirection

    nothing
end

function main()
    MPI.Init()

    size_in = DATA_DIMS
    test_transform_types(size_in)

    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)

    # Let MPI_Dims_create choose the 2D decomposition.
    pdims_2d = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    test_pencil_plans(size_in, pdims_2d)

    MPI.Finalize()
end

main()
