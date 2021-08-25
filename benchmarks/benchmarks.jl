#!/usr/bin/env julia

using PencilFFTs
using PencilFFTs.PencilArrays

import FFTW
using MPI
import Pkg

using OrderedCollections: OrderedDict
using TimerOutputs

using LinearAlgebra
using Printf
using Profile
using Random

TimerOutputs.enable_debug_timings(PencilFFTs)
TimerOutputs.enable_debug_timings(PencilArrays)
TimerOutputs.enable_debug_timings(Transpositions)

FFTW.set_num_threads(1)

const PROFILE = false
const PROFILE_OUTPUT = "profile.txt"
const PROFILE_DEPTH = 8

const MEASURE_GATHER = false

const DIMS_DEFAULT = "32,64,128"

const DEV_NULL = @static Sys.iswindows() ? "nul" : "/dev/null"

const SEPARATOR = string("\n", "*"^80)

const RESULTS_HEADER =
"""# The last 4 columns show timing statistics (mean/std/min/max) in milliseconds.
    #
    # The last two letters in the name of this file determine the PencilFFTs
    # parameters used in the benchmarks.
    #
    # The first letter indicates whether dimension permutations are performed:
    #
    #     P = permutations (default in PencilFFTs)
    #     N = no permutations
    #
    # The second letter indicates the MPI transposition method:
    #
    #     I = Isend/Irecv (default in PencilFFTs)
    #     A = Alltoallv
    #
    """

mutable struct TimerData
    avg :: Float64
    std :: Float64
    min :: Float64
    max :: Float64
    TimerData() = new(0, 0, Inf, -1)
end

function Base.:*(t::TimerData, v)
    t.avg *= v
    t.std *= v
    t.min *= v
    t.max *= v
    t
end

function getenv(::Type{T}, key, default = nothing) where {T}
    s = get(ENV, key, nothing)
    if s === nothing
        default
    elseif T <: AbstractString
        s
    else
        parse(T, s)
    end
end

getenv(key, default::T) where {T} = getenv(T, key, default)

function parse_params()
    dims_str = getenv("PENCILFFTS_BENCH_DIMENSIONS", DIMS_DEFAULT)
    repetitions = getenv("PENCILFFTS_BENCH_REPETITIONS", 100)
    outfile = getenv(String, "PENCILFFTS_BENCH_OUTPUT", nothing)
    (
        dims = parse_dimensions(dims_str) :: Dims{3},
        iterations = repetitions :: Int,
        outfile = outfile :: Union{Nothing,String},
    )
end

# Slab decomposition
function create_pencils(topo::MPITopology{1}, data_dims, permutation::Val{true};
                        kwargs...)
    pen1 = Pencil(topo, data_dims, (2, ); kwargs...)
    pen2 = Pencil(pen1, decomp_dims=(3, ), permute=Permutation(2, 1, 3); kwargs...)
    pen3 = Pencil(pen2, decomp_dims=(2, ), permute=Permutation(3, 2, 1); kwargs...)
    pen1, pen2, pen3
end

function create_pencils(topo::MPITopology{1}, data_dims,
                        permutation::Val{false}; kwargs...)
    pen1 = Pencil(topo, data_dims, (2, ); kwargs...)
    pen2 = Pencil(pen1, decomp_dims=(3, ); kwargs...)
    pen3 = Pencil(pen2, decomp_dims=(2, ); kwargs...)
    pen1, pen2, pen3
end

# Pencil decomposition
function create_pencils(topo::MPITopology{2}, data_dims, permutation::Val{true};
                        kwargs...)
    pen1 = Pencil(topo, data_dims, (2, 3); kwargs...)
    pen2 = Pencil(pen1, decomp_dims=(1, 3), permute=Permutation(2, 1, 3); kwargs...)
    pen3 = Pencil(pen2, decomp_dims=(1, 2), permute=Permutation(3, 2, 1); kwargs...)
    pen1, pen2, pen3
end

function create_pencils(topo::MPITopology{2}, data_dims, permutation::Val{false};
                        kwargs...)
    pen1 = Pencil(topo, data_dims, (2, 3); kwargs...)
    pen2 = Pencil(pen1, decomp_dims=(1, 3); kwargs...)
    pen3 = Pencil(pen2, decomp_dims=(1, 2); kwargs...)
    pen1, pen2, pen3
end

function benchmark_pencils(comm, proc_dims::Tuple, data_dims::Tuple;
                           iterations=1,
                           with_permutations::Val=Val(true),
                           extra_dims::Tuple=(),
                           transpose_method=Transpositions.PointToPoint(),
                          )
    topo = MPITopology(comm, proc_dims)
    M = length(proc_dims)
    @assert M in (1, 2)

    to = TimerOutput()

    pens = create_pencils(topo, data_dims, with_permutations, timer=to)

    u = map(p -> PencilArray{Float64}(undef, p, extra_dims...), pens)

    myrank = MPI.Comm_rank(comm)
    rng = MersenneTwister(42 + myrank)
    randn!(rng, u[1])
    u[1] .+= 10 * myrank
    u_orig = copy(u[1])

    transpose_m!(a, b) = transpose!(a, b, method=transpose_method)

    # Precompile functions
    transpose_m!(u[2], u[1])
    transpose_m!(u[3], u[2])
    transpose_m!(u[2], u[3])
    transpose_m!(u[1], u[2])
    gather(u[2])

    reset_timer!(to)

    for it = 1:iterations
        transpose_m!(u[2], u[1])
        transpose_m!(u[3], u[2])
        transpose_m!(u[2], u[3])
        transpose_m!(u[1], u[2])
        MEASURE_GATHER && gather(u[2])
    end

    @assert u[1] == u_orig

    if myrank == 0
        println("\n",
            """
            Processes:               $proc_dims
            Data dimensions:         $data_dims $(isempty(extra_dims) ? "" : "× $extra_dims")
            Permutations (1, 2, 3):  $(permutation.(pens))
            Transpositions:          1 -> 2 -> 3 -> 2 -> 1
            Method:                  $(transpose_method)
            """)
        println(to, SEPARATOR)
    end

    if PROFILE
        Profile.clear()
        @profile for it = 1:iterations
            transpose_m!(u[2], u[1])
            transpose_m!(u[3], u[2])
            transpose_m!(u[2], u[3])
            transpose_m!(u[1], u[2])
        end
        if myrank == 0
            open(io -> Profile.print(io, maxdepth=PROFILE_DEPTH),
                 PROFILE_OUTPUT, "w")
        end
    end

    nothing
end

function benchmark_rfft(comm, proc_dims::Tuple, data_dims::Tuple;
                        extra_dims=(),
                        iterations=1,
                        transpose_method=Transpositions.PointToPoint(),
                        permute_dims=Val(true),
                       )
    isroot = MPI.Comm_rank(comm) == 0

    to = TimerOutput()
    plan = PencilFFTPlan(data_dims, Transforms.RFFT(), proc_dims, comm,
                         extra_dims=extra_dims,
                         permute_dims=permute_dims,
                         fftw_flags=FFTW.ESTIMATE,
                         timer=to, transpose_method=transpose_method)

    if isroot
        println("\n", plan, "\nMethod: ", plan.transpose_method)
        println("Permutations: $permute_dims\n")
    end

    u = allocate_input(plan)
    v = allocate_output(plan)
    uprime = similar(u)

    randn!(u)

    # Warm-up
    mul!(v, plan, u)
    ldiv!(uprime, plan, v)

    @assert u ≈ uprime

    reset_timer!(to)

    t = TimerData()

    for it = 1:iterations
        τ = -MPI.Wtime()
        mul!(v, plan, u)
        ldiv!(u, plan, v)
        τ += MPI.Wtime()
        t.avg += τ
        t.std += τ^2
        t.min = min(τ, t.min)
        t.max = max(τ, t.max)
    end

    t.avg /= iterations
    t.std = sqrt(t.std / iterations - t.avg^2)

    t *= 1000  # in milliseconds

    events = (to["PencilFFTs mul!"], to["PencilFFTs ldiv!"])
    @assert all(TimerOutputs.ncalls.(events) .== iterations)
    t_to = sum(TimerOutputs.time.(events)) / iterations / 1e6  # in milliseconds

    if isroot
        @printf("Average time: %.8f ms (TimerOutputs) over %d repetitions\n",
                t_to, iterations)
        @printf("              %.8f ms (MPI_Wtime) ± %.8f ms \n\n",
                t.avg, t.std / 2)
        print_timers(to, iterations, transpose_method)
        println(SEPARATOR)
    end

    t
end

struct AggregatedTimes{TM}
    transpose :: TM
    mpi  :: Float64  # MPI time in μs
    fft  :: Float64  # FFTs in μs
    data :: Float64  # data copies in μs
    others :: Float64
end

function AggregatedTimes(to::TimerOutput, transpose_method)
    repetitions = TimerOutputs.ncalls(to)

    avgtime(x) = TimerOutputs.time(x) / repetitions
    avgtime(::Nothing) = 0.0

    fft = avgtime(to["FFT"])

    tf = TimerOutputs.flatten(to)
    data = avgtime(tf["copy_permuted!"]) + avgtime(tf["copy_range!"])

    mpi = if transpose_method === Transpositions.PointToPoint()
        t = avgtime(to["MPI.Waitall!"])
        if haskey(tf, "wait receive")  # this will be false in serial mode
            t += avgtime(tf["wait receive"])
        end
        t
    elseif transpose_method === Transpositions.Alltoallv()
        avgtime(tf["MPI.Alltoallv!"]) + avgtime(to["MPI.Waitall!"])
    end

    others = 0.0
    if haskey(to, "normalise")  # normalisation of inverse transform
        others += avgtime(to["normalise"])
    end

    let scale = 1e-6  # convert to ms
        mpi *= scale / 2   # 2 transposes per iteration
        fft *= scale / 3   # 3 FFTs per iteration
        data *= scale / 2
        others *= scale
    end

    AggregatedTimes(transpose_method, mpi, fft, data, others)
end

# 2 transpositions + 3 FFTs
time_total(t::AggregatedTimes) = 2 * (t.mpi + t.data) + 3 * t.fft + t.others

function Base.show(io::IO, t::AggregatedTimes)
    maybe_newline = ""
    for p in (string(t.transpose) => t.mpi,
              "FFT" => t.fft, "(un)pack" => t.data, "others" => t.others)
        @printf io "%s  Average %-10s = %.6f ms" maybe_newline p.first p.second
        maybe_newline = "\n"
    end
    io
end

function print_timers(to::TimerOutput, iterations, transpose_method)
    println(to, "\n")

    @assert TimerOutputs.ncalls(to["PencilFFTs mul!"]) == iterations

    t_fw = AggregatedTimes(to["PencilFFTs mul!"], transpose_method)
    t_bw = AggregatedTimes(to["PencilFFTs ldiv!"], transpose_method)

    println("Forward transforms\n", t_fw)
    println("\nBackward transforms\n", t_bw)

    t_all_measured = sum(time_total.((t_fw, t_bw)))  # in milliseconds

    # Actual time taken by parallel FFTs.
    t_all = TimerOutputs.tottime(to) / 1e6 / iterations

    # Fraction the elapsed time that is not included in t_all_measured.
    t_missing = t_all - t_all_measured
    percent_missing = (1 - t_all_measured / t_all) * 100

    @printf("\nTotal from timers: %.4f ms/iteration (%.4f ms / %.2f%% missing)\n",
            t_all_measured, t_missing, percent_missing)

    nothing
end

function parse_dimensions(arg::AbstractString) :: Dims{3}
    ints = try
        sp = split(arg, ',')
        parse.(Int, sp)
    catch e
        error("Could not parse dimensions from '$arg'")
    end
    if length(ints) == 1
        N = ints[1]
        (N, N, N)
    elseif length(ints) == 3
        ntuple(n -> ints[n], Val(3))
    else
        error("Incorrect number of dimensions in '$ints'")
    end
end

make_index(::Val{true}) = 1
make_index(::Val{false}) = 2

make_index(::Transpositions.PointToPoint) = 1
make_index(::Transpositions.Alltoallv) = 2

make_index(stuff...) = CartesianIndex(make_index.(stuff))

function run_benchmarks(params)
    comm = MPI.COMM_WORLD
    Nproc = MPI.Comm_size(comm)
    myrank = MPI.Comm_rank(comm)

    dims = params.dims
    iterations = params.iterations
    outfile = params.outfile

    if myrank == 0
        @info "Global dimensions: $dims"
        @info "Repetitions:       $iterations"
    end

    # Let MPI_Dims_create choose the decomposition.
    proc_dims = let pdims = zeros(Int, 2)
        MPI.Dims_create!(Nproc, pdims)
        pdims[1], pdims[2]
    end

    transpose_methods = (Transpositions.PointToPoint(),
                         Transpositions.Alltoallv())
    permutes = (Val(true), Val(false))
    timings = Array{TimerData}(undef, 2, 2)

    map(Iterators.product(transpose_methods, permutes)) do (method, permute)
        I = make_index(permute, method)
        timings[I] = benchmark_rfft(
            comm, proc_dims, dims;
            iterations = iterations, permute_dims = permute, transpose_method = method,
        )
    end

    columns = OrderedDict{String,Union{Int,Float64}}(
        "(1) Nx" => dims[1],
        "(2) Ny" => dims[2],
        "(3) Nz" => dims[3],
        "(4) num_procs" => Nproc,
        "(5) P1" => proc_dims[1],
        "(6) P2" => proc_dims[2],
        "(7) repetitions" => iterations,
    )

    cases = (
        :PI => make_index(Val(true), Transpositions.PointToPoint()),
        :PA => make_index(Val(true), Transpositions.Alltoallv()),
        :NI => make_index(Val(false), Transpositions.PointToPoint()),
        :NA => make_index(Val(false), Transpositions.Alltoallv()),
    )

    if myrank == 0 && outfile !== nothing
        for (name, ind) in cases
            a, b = splitext(outfile)
            fname = string(a, "_", name, b)
            write_results(fname, columns, timings[ind])
        end
    end

    nothing
end

function write_results(outfile, columns, t)
    @info "Writing to $outfile"
    newfile = !isfile(outfile)
    open(outfile, "a") do io
        if newfile
            print(io, RESULTS_HEADER, "#")
            n = length(columns)
            mkname(c, name) = "($(n + c)) $name"
            names = Iterators.flatten((
                keys(columns),
                (mkname(1, "mean"), mkname(2, "std"),
                 mkname(3, "min"), mkname(4, "max"))
            ))
            for name in names
                print(io, "  ", name)
            end
            println(io)
        end
        vals = Iterators.flatten((values(columns), t.avg, t.std, t.min, t.max))
        for val in vals
            print(io, "  ", val)
        end
        println(io)
    end
    nothing
end

MPI.Init()
if MPI.Comm_rank(MPI.COMM_WORLD) == 0
    Pkg.status(mode = Pkg.PKGMODE_MANIFEST)
end
params = parse_params()
run_benchmarks(params)
