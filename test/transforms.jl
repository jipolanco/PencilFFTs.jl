#!/usr/bin/env julia

using PencilFFTs

import FFTW
using MPI

using InteractiveUtils
using LinearAlgebra
using Random
using Test
using TimerOutputs

include("include/MPITools.jl")
using .MPITools

const DATA_DIMS = (24, 12, 6)

const FAST_TESTS = !("--all" in ARGS)

# Only test a few r2r kinds in "fast" tests.
# Otherwise, test all of them.
const TEST_KINDS_R2R = FAST_TESTS ?
    (FFTW.REDFT00, FFTW.REDFT01) : Transforms.R2R_SUPPORTED_KINDS

function test_transform_types(size_in)
    @testset "r2c transforms" begin
        transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
        fft_params = PencilFFTs.GlobalFFTParams(size_in, transforms)

        @test fft_params isa PencilFFTs.GlobalFFTParams{Float64, 3, false,
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

    @testset "NoTransform" begin
        transform = Transforms.NoTransform()
        transform! = Transforms.NoTransform!()

        @test binv(transform) === transform
        @test binv(transform!) === transform!

        x = zeros(4)
        p = Transforms.plan(transform, x)
        p! = Transforms.plan(transform!, x)

        @test p * x !== x  # creates copy
        @test p * x == x
        @test p! * x === x

        y = similar(x)
        @test mul!(y, p, x) === y == x
        @test mul!(x, p, x) === x  # this is also allowed

        rand!(x)
        @test ldiv!(x, p, y) === x == y

        # in-place IdentityPlan applied to out-of-place data
        @test_throws ArgumentError mul!(y, p!, x)
        @test mul!(x, p, x) === x
        @test ldiv!(x, p, x) === x
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
        end
    end

    @testset "In-place transforms 1D" begin
        FFT = Transforms.FFT()
        FFT! = Transforms.FFT!()

        @inferred Transforms.is_inplace(FFT, FFT, FFT!)
        @inferred Transforms.is_inplace(FFT!, FFT, FFT)
        @inferred Transforms.is_inplace(FFT, FFT, FFT)
        @inferred Transforms.is_inplace(FFT!, FFT!, FFT!)

        @test Transforms.is_inplace(FFT, FFT, FFT!) === nothing
        @test Transforms.is_inplace(FFT, FFT!, FFT) === nothing
        @test Transforms.is_inplace(FFT, FFT!, FFT!) === nothing
        @test Transforms.is_inplace(FFT!, FFT, FFT!) === nothing
        @test Transforms.is_inplace(FFT!, FFT!, FFT!) === true
        @test Transforms.is_inplace(FFT, FFT, FFT) === false

        @inferred PencilFFTs.GlobalFFTParams(size_in, (FFT!, FFT!, FFT!))

        # Cannot combine in-place and out-of-place transforms.
        @test_throws ArgumentError PencilFFTs.GlobalFFTParams(size_in, (FFT, FFT!, FFT!))
    end

    nothing
end

function test_inplace(::Type{T}, comm, proc_dims, size_in;
                      extra_dims=()) where {T}
    transforms = Transforms.FFT!()  # in-place c2c FFT
    plan = PencilFFTPlan(size_in, transforms, proc_dims, comm, T;
                         extra_dims=extra_dims)

    # Out-of-place plan, just for verifying that we throw errors.
    plan_oop = PencilFFTPlan(size_in, Transforms.FFT(), proc_dims, comm, T;
                             extra_dims=extra_dims)

    dims_fft = 1:length(size_in)

    @testset "In-place transforms 3D" begin
        test_transform(plan, x -> FFTW.plan_fft!(x, dims_fft), plan_oop)
    end

    nothing
end

function test_transform(plan::PencilFFTPlan, args...; kwargs...)

    println("\n", "-"^60, "\n\n", plan, "\n")

    @inferred allocate_input(plan)
    @inferred allocate_input(plan, 2, 3)
    @inferred allocate_input(plan, Val(3))
    @inferred allocate_output(plan)

    test_transform(Val(is_inplace(plan)), plan, args...; kwargs...)
end

function test_transform(inplace::Val{true}, plan::PencilFFTPlan,
                        serial_planner::Function, plan_oop::PencilFFTPlan;
                        root=0)
    @assert !is_inplace(plan_oop)

    vi = allocate_input(plan)
    @test vi isa PencilArrays.ManyPencilArray

    let vo = allocate_output(plan)
        @test typeof(vi) === typeof(vo)
    end

    u = first(vi)  # input PencilArray
    v = last(vi)   # output PencilArray

    randn!(u)
    u_initial = copy(u)
    ug = gather(u, root)  # for comparison with serial FFT

    # Input array type ManyPencilArray{...} incompatible with out-of-place
    # plans.
    @test_throws ArgumentError plan_oop * vi

    # Out-of-place plan applied to in-place data.
    @test_throws ArgumentError mul!(v, plan_oop, u)

    let vi_other = allocate_input(plan)
        # Input and output arrays for in-place plan must be the same.
        @test_throws ArgumentError mul!(vi_other, plan, vi)
    end

    # Input array type incompatible with in-place plans.
    @test_throws ArgumentError plan * u
    @test_throws ArgumentError plan \ v

    @assert PencilFFTs.is_inplace(plan)
    plan * vi               # apply in-place forward transform
    @test isempty(u) || !(u ≈ u_initial)  # `u` was modified!

    # Now `v` contains the transformed data.
    vg = gather(v, root)
    if ug !== nothing && vg !== nothing
        p = serial_planner(ug)
        p * ug  # apply serial in-place FFT
        @test ug ≈ vg
    end

    plan \ vi  # apply in-place backward transform

    # Now `u` contains the initial data (approximately).
    @test u ≈ u_initial

    ug_again = gather(u, root)
    if ug !== nothing && ug_again !== nothing
        p \ ug  # apply serial in-place FFT
        @test ug ≈ ug_again
    end

    let components = ((Val(3), ), (3, 2))
        @testset "Components: $comp" for comp in components
            vi = allocate_input(plan, comp...)
            u = first.(vi)
            v = last.(vi)
            randn!.(u)
            u_initial = copy.(u)

            # In some cases, generally when data is split among too many
            # processes, the local process may have no data.
            empty = isempty(first(u))

            plan * vi
            @test empty || !all(u_initial .≈ u)

            plan \ vi
            @test all(u_initial .≈ u)
        end
    end

    nothing
end

function test_transform(inplace::Val{false}, plan::PencilFFTPlan,
                        serial_planner::Function; root=0)
    u = allocate_input(plan)
    v = allocate_output(plan)

    @test u isa PencilArray

    randn!(u)

    mul!(v, plan, u)
    uprime = similar(u)
    ldiv!(uprime, plan, v)

    @test u ≈ uprime

    # Compare result with serial FFT.
    ug = gather(u, root)
    vg = gather(v, root)

    if ug !== nothing && vg !== nothing
        let p = serial_planner(ug)
            vg_serial = p * ug
            @test vg ≈ vg_serial
        end
    end

    nothing
end

function test_transforms(::Type{T}, comm, proc_dims, size_in;
                         extra_dims=()) where {T}
    plan_kw = (:extra_dims => extra_dims, )
    N = length(size_in)

    make_plan(planner, args...; dims=1:N) = x -> planner(x, args..., dims)

    pair_r2r(tr::Transforms.R2R) =
        tr => make_plan(FFTW.plan_r2r, Transforms.kind(tr))
    pairs_r2r = (pair_r2r(Transforms.R2R{k}()) for k in TEST_KINDS_R2R)

    pairs = if FAST_TESTS &&
            (T === Float32 || !isempty(extra_dims) || length(proc_dims) == 1)
        # Only test one transform with Float32 / extra_dims / 1D decomposition.
        (Transforms.RFFT() => make_plan(FFTW.plan_rfft), )
    else
        (
         Transforms.FFT() => make_plan(FFTW.plan_fft),
         Transforms.RFFT() => make_plan(FFTW.plan_rfft),
         Transforms.BFFT() => make_plan(FFTW.plan_bfft),
         Transforms.NoTransform() => (x -> Transforms.IdentityPlan()),
         pairs_r2r...,
         (Transforms.NoTransform(), Transforms.RFFT(), Transforms.FFT())
             => make_plan(FFTW.plan_rfft, dims=2:3),
         (Transforms.FFT(), Transforms.NoTransform(), Transforms.FFT())
             => make_plan(FFTW.plan_fft, dims=(1, 3)),
         (Transforms.FFT(), Transforms.NoTransform(), Transforms.NoTransform())
             => make_plan(FFTW.plan_fft, dims=1),
         Transforms.BRFFT() => make_plan(FFTW.plan_brfft),  # not yet supported
        )
    end

    @testset "$(p.first) -- $T" for p in pairs
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
        test_transform(plan, fftw_planner)
    end

    nothing
end

function test_pencil_plans(size_in::Tuple, pdims::Tuple, comm)
    @assert length(size_in) >= 3

    @inferred PencilFFTPlan(size_in, Transforms.RFFT(), pdims, comm, Float64)

    @testset "Transform types" begin
        let transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
            @inferred PencilFFTPlan(size_in, transforms, pdims, comm)
            @inferred PencilFFTs._input_data_type(Float64, transforms...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.FFT())
            @test PencilFFTs._input_data_type(Float32, transforms...) ===
                ComplexF32
            @inferred PencilFFTs._input_data_type(Float32, transforms...)
        end

        let transforms = (Transforms.NoTransform(), Transforms.NoTransform())
            @test PencilFFTs._input_data_type(Float32, transforms...) ===
                Float32
            @inferred PencilFFTs._input_data_type(Float32, transforms...)
        end
    end

    if FAST_TESTS && length(pdims) == 1
        # Only test one case for 1D decomposition.
        types = (Float64, )
        extra_dims = ((), )
    else
        types = (Float64, Float32)
        extra_dims = ((), (3, ))
    end

    for T in types, edims in extra_dims
        test_inplace(T, comm, pdims, size_in, extra_dims=edims)
        test_transforms(T, comm, pdims, size_in, extra_dims=edims)
    end

    nothing
end

function main()
    MPI.Init()

    size_in = DATA_DIMS
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    silence_stdout(comm)

    test_transform_types(size_in)

    pdims_1d = (Nproc, )  # 1D ("slab") decomposition

    # Let MPI_Dims_create choose the 2D decomposition.
    pdims_2d = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    for p in (pdims_1d, pdims_2d)
        test_pencil_plans(size_in, p, comm)
    end

    MPI.Finalize()
end

main()
