using PencilFFTs
using .Transforms: binv, is_inplace

import FFTW
using MPI

using LinearAlgebra
using Random
using Test
using TimerOutputs

include("include/MPITools.jl")
using .MPITools

const DATA_DIMS = (16, 12, 6)

const FAST_TESTS = !("--all" in ARGS)

# Test all possible r2r transforms.
const TEST_KINDS_R2R = Transforms.R2R_SUPPORTED_KINDS

function test_transform_types(size_in)
    @testset "r2c transforms" begin
        transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
        fft_params = PencilFFTs.GlobalFFTParams(size_in, transforms)

        @test fft_params isa PencilFFTs.GlobalFFTParams{Float64, 3, false,
                                                        typeof(transforms)}
        @test binv(Transforms.RFFT(), 10) === Transforms.BRFFT(10)
        @test binv(Transforms.RFFT(), 11) === Transforms.BRFFT(11)

        transforms_binv = binv.(transforms, size_in)
        size_out = Transforms.length_output.(transforms, size_in)

        @test transforms_binv ===
            (Transforms.BRFFT(size_in[1]), Transforms.BFFT(), Transforms.BFFT())
        @test size_out === (size_in[1] ÷ 2 + 1, size_in[2:end]...)
        @test Transforms.length_output.(transforms_binv, size_out) === size_in

        @test PencilFFTs.input_data_type(fft_params) === Float64
    end

    @testset "c2c transforms" begin
        n = 4
        @test binv(Transforms.FFT(), n) === Transforms.BFFT()
        @test binv(Transforms.BFFT(), n) === Transforms.FFT()
        @test binv(Transforms.FFT!(), n) === Transforms.BFFT!()
        @test binv(Transforms.BFFT!(), n) === Transforms.FFT!()
    end

    @testset "NoTransform" begin
        transform = Transforms.NoTransform()
        transform! = Transforms.NoTransform!()

        @test binv(transform, 42) === transform
        @test binv(transform!, 42) === transform!

        @test !is_inplace(transform)
        @test is_inplace(transform!)

        x = rand(4)
        p = Transforms.plan(transform, x)
        p! = Transforms.plan(transform!, x)

        @test p * x !== x  # creates copy
        @test p * x == x
        @test p \ x !== x  # creates copy
        @test p \ x == x

        @test p! * x === x
        @test p! \ x === x

        y = similar(x)
        @test mul!(y, p, x) === y == x
        @test mul!(x, p, x) === x  # this is also allowed

        rand!(x)
        @test ldiv!(x, p, y) === x == y

        # in-place IdentityPlan applied to out-of-place data
        @test_throws ArgumentError mul!(y, p!, x)
        @test mul!(x, p!, x) === x
        @test ldiv!(x, p!, x) === x
        @test mul!(x, p, x) === x
        @test ldiv!(x, p, x) === x
    end

    # Test type stability of generated plan_r2r (which, as defined in FFTW.jl,
    # is type unstable!). See comments of `plan` in src/Transforms/r2r.jl.
    @testset "r2r transforms" begin
        kind = FFTW.REDFT01
        transform = Transforms.R2R(kind)
        transform! = Transforms.R2R!(kind)

        @inferred (() -> Transforms.R2R(FFTW.REDFT10))()
        @inferred (() -> Transforms.R2R!(FFTW.REDFT10))()

        let kind_inv = FFTW.REDFT10
            @test binv(transform, 42) === Transforms.R2R(kind_inv)
            @test binv(transform!, 42) === Transforms.R2R!(kind_inv)
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
            @test_throws ArgumentError T(kind)
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

    @testset "Transforms internals" begin
        FFT = Transforms.FFT()
        x = zeros(ComplexF32, 3, 4)
        @test Transforms.scale_factor(FFT, x) == length(x)
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

function test_transform(
        inplace::Val{false}, plan::PencilFFTPlan, serial_planner;
        root = 0, is_c2r = false,
    )
    u = allocate_input(plan)
    v = allocate_output(plan)

    @test u isa PencilArray

    if is_c2r
        # Generate input via inverse r2c transform, to avoid issues with input
        # not respecting required symmetry.
        randn!(v)
        ldiv!(u, plan, v)
        fill!(v, 0)
    else
        randn!(u)
    end

    mul!(v, plan, u)
    uprime = similar(u)
    ldiv!(uprime, plan, v)

    @test u ≈ uprime

    # Compare result with serial FFT.
    ug = gather(u, root)
    vg = gather(v, root)

    if ug !== nothing && vg !== nothing && serial_planner !== nothing
        let p = serial_planner(ug)
            vg_serial = p * ug
            if is_c2r
                @test size(vg) != size(vg_serial)
            else
                @test vg ≈ vg_serial
            end
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
    pairs_r2r = (pair_r2r(Transforms.R2R(k)) for k in TEST_KINDS_R2R)

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

         # TODO compare BRFFT with serial equivalent?
         # The special case of BRFFT is a bit complicated, because
         # multidimensional FFTW plans returned by `plan_brfft` perform the
         # actual c2r transform along the first dimension. In PencilFFTs we do
         # the opposite: the c2r transform is applied along the last dimension.
         Transforms.BRFFT(size_in) => nothing,
        )
    end

    @testset "$(p.first) -- $T" for p in pairs
        transform, fftw_planner = p
        is_c2r = transform isa Transforms.BRFFT
        plan = @inferred PencilFFTPlan(
            size_in, transform, proc_dims, comm, T; plan_kw...)
        test_transform(plan, fftw_planner; is_c2r)
    end

    nothing
end

function test_pencil_plans(size_in::Tuple, pdims::Tuple, comm)
    @assert length(size_in) >= 3

    @inferred PencilFFTPlan(size_in, Transforms.RFFT(), pdims, comm, Float64)

    let to = TimerOutput()
        plan = PencilFFTPlan(size_in, Transforms.RFFT(), pdims, comm, Float64,
                             timer=to)
        @test timer(plan) === to
    end

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

# Test N-dimensional transforms decomposing along M dimensions.
function test_dimensionality(dims::Dims{N}, ::Val{M}, comm;
                             plan_kw...) where {N, M}
    @assert M < N
    pdims = make_pdims(Val(M), MPI.Comm_size(comm))

    @testset "Decompose $M/$N dims" begin
        # Out-of-place transform.
        let transform = Transforms.RFFT()
            plan = PencilFFTPlan(dims, transform, pdims, comm; plan_kw...)
            test_transform(plan, FFTW.plan_rfft)
        end

        # In-place transform.
        let transform = Transforms.FFT!()
            plan = PencilFFTPlan(dims, transform, pdims, comm; plan_kw...)
            plan_oop = PencilFFTPlan(dims, Transforms.FFT(), pdims, comm;
                                     plan_kw...)
            test_transform(plan, FFTW.plan_fft!, plan_oop)
        end

    end

    nothing
end

function test_dimensionality(comm)
    # 1D decomposition of 2D problem.
    test_dimensionality((11, 15), Val(1), comm)

    let dims = (11, 7, 13)
        test_dimensionality(dims, Val(1), comm)  # slab decomposition
        test_dimensionality(dims, Val(2), comm)  # pencil decomposition
    end

    let dims = (9, 7, 5, 12)
        test_dimensionality(dims, Val(1), comm)
        test_dimensionality(dims, Val(2), comm)
        test_dimensionality(dims, Val(3), comm)  # 3D decomposition of 4D problem

        # Same with some non-default options for the plans.
        test_dimensionality(
            dims, Val(3), comm,
            permute_dims=Val(false),
            transpose_method=Transpositions.Alltoallv(),
        )
    end

    nothing
end

# Test incompatibilities between plans and inputs.
function test_incompatibility(comm)
    pdims = (MPI.Comm_size(comm), )
    dims = (10, 8)
    dims_other = (6, 8)

    @testset "Incompatibility" begin
        plan = PencilFFTPlan(dims, Transforms.FFT(), pdims, comm)
        plan! = PencilFFTPlan(dims, Transforms.FFT!(), pdims, comm)

        u = allocate_input(plan)  :: PencilArray
        v = allocate_output(plan) :: PencilArray

        # "input array type PencilArray{...} incompatible with in-place plans"
        @test_throws ArgumentError plan! * u

        # "input array type ManyPencilArray{...} incompatible with out-of-place plans"
        M! = allocate_input(plan!) :: ManyPencilArray
        @test_throws ArgumentError plan * M!

        # "array types (...) incompatible with in-place plans"
        @test_throws ArgumentError mul!(v, plan!, u)

        # "array types (...) incompatible with out-of-place plan"
        @test_throws ArgumentError mul!(M!, plan, M!)

        # "out-of-place plan applied to aliased data"
        @test_throws ArgumentError mul!(last(M!), plan, first(M!))

        # "collections have different lengths: 3 ≠ 2"
        u3 = allocate_input(plan, Val(3))
        v2 = allocate_output(plan, Val(2))
        @test_throws ArgumentError mul!(v2, plan, u3)

        let plan_other = PencilFFTPlan(dims_other, Transforms.FFT(), pdims, comm)
            # "unexpected dimensions of input data"
            @test_throws ArgumentError plan_other * u

            # "unexpected dimensions of output data"
            v_other = allocate_output(plan_other)
            @test_throws ArgumentError mul!(v_other, plan, u)
        end
    end

    nothing
end

function make_pdims(::Val{M}, Nproc) where {M}
    # Let MPI.Dims_create! choose the decomposition.
    pdims = zeros(Int, M)
    MPI.Dims_create!(Nproc, pdims)
    ntuple(d -> pdims[d], Val(M))
end

function main()
    size_in = DATA_DIMS
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    silence_stdout(comm)

    test_transform_types(size_in)
    test_incompatibility(comm)
    test_dimensionality(comm)

    pdims_1d = (Nproc, )  # 1D ("slab") decomposition
    pdims_2d = make_pdims(Val(2), Nproc)

    for p in (pdims_1d, pdims_2d)
        test_pencil_plans(size_in, p, comm)
    end

    nothing
end

MPI.Initialized() || MPI.Init()
main()
