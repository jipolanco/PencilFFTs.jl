const FFTWPlanOrIdentity = Union{FFTW.FFTWPlan, Transforms.IdentityPlan}
const ValBool = Union{Val{false}, Val{true}}

# One-dimensional distributed FFT plan.
struct PencilPlan1D{Pi <: Pencil,
                    Po <: Pencil,
                    Tr <: AbstractTransform,
                    FFTPlanF <: FFTWPlanOrIdentity,
                    FFTPlanB <: FFTWPlanOrIdentity,
                   }
    # Each pencil pair describes the decomposition of input and output FFT
    # data. The two pencils will be different for transforms that do not
    # preserve the size and element type of the data (e.g. real-to-complex
    # transforms). Otherwise, they will be typically identical.
    pencil_in  :: Pi       # pencil before transform
    pencil_out :: Po       # pencil after transform
    transform  :: Tr       # transform type

    fft_plan   :: FFTPlanF  # forward FFTW plan
    bfft_plan  :: FFTPlanB  # backward FFTW plan (unnormalised)

    scale_factor :: Int  # scale factor for backward transform
end

"""
    PencilFFTPlan{T,N,M}

Plan for N-dimensional FFT-based transform on MPI-distributed data.

---

    PencilFFTPlan(size_global::Dims{N}, transforms,
                  proc_dims::Dims{M}, comm::MPI.Comm, [real_type=Float64];
                  fftw_flags=FFTW.ESTIMATE, fftw_timelimit=FFTW.NO_TIMELIMIT,
                  permute_dims=Val(true),
                  transpose_method=TransposeMethods.IsendIrecv(),
                  timer=TimerOutput(),
                  )

Create plan for N-dimensional transform.

`size_global` specifies the global dimensions of the input data.

`transforms` should be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[`Transform types`](@ref Transforms). Alternatively, `transforms` may be a
single transform that will be automatically expanded into `N` equivalent
transforms. This is illustrated in the example below.

The transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms. For multidimensional transforms of
real data, this means that a real-to-complex transform must be performed along
the first dimension, and then complex-to-complex transforms are performed
along the other two dimensions (see example below).

The data is distributed over the MPI processes in the `comm` communicator.
The distribution is performed over `M` dimensions (with `M < N`) according to
the values in `proc_dims`, which specifies the number of MPI processes to put
along each dimension.

# Optional arguments

- The floating point precision can be selected by setting `real_type` parameter,
  which is `Float64` by default.

- The keyword arguments `fftw_flags` and `fftw_timelimit` are passed to the
  `FFTW` plan creation functions (see [`AbstractFFTs`
  docs](https://juliamath.github.io/AbstractFFTs.jl/stable/api/#AbstractFFTs.plan_fft)).

- `permute_dims` determines whether the indices of the output data should be
  reversed. For instance, if the input data has global dimensions
  `(Nx, Ny, Nz)`, then the output of a complex-to-complex FFT would have
  dimensions `(Nz, Ny, Nx)`. This enables FFTs to always be performed along
  the first (i.e. fastest) array dimension, which could lead to performance
  gains. This option is enabled by default. For type inference reasons, it must
  be a value type (`Val(true)` or `Val(false)`).

- `transpose_method` allows to select between implementations of the global
  data transpositions. See the [`transpose!`](@ref) docs for details.

- `timer` should be a `TimerOutput` object.
  See [Measuring performance](@ref Pencils.measuring_performance) for details.

# Example

Suppose we want to perform a 3D transform of real data. The data is to be
decomposed along two dimensions, over 8 MPI processes:

```julia
size_global = (64, 32, 128)  # size of real input data

# Perform real-to-complex transform along the first dimension, then
# complex-to-complex transforms along the other dimensions.
transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
# transforms = Transforms.RFFT()  # this is equivalent to the above line

proc_dims = (4, 2)  # 2D decomposition
comm = MPI.COMM_WORLD

plan = PencilFFTPlan(size_global, transforms, proc_dims, comm)
```

"""
struct PencilFFTPlan{T,
                     N,
                     M,
                     G <: GlobalFFTParams,
                     P <: NTuple{N, PencilPlan1D},
                     TransposeMethod <: AbstractTransposeMethod,
                    }
    global_params :: G
    topology      :: MPITopology{M}

    # One-dimensional plans, including data decomposition configurations.
    plans :: P

    # Scale factor to be applied after backwards transforms.
    scale_factor :: Int

    # `method` parameter passed to `Pencils.transpose!`
    transpose_method :: TransposeMethod

    # Temporary data buffers.
    ibuf :: Vector{UInt8}
    obuf :: Vector{UInt8}

    # Runtime timing.
    # Should be used along with the @timeit_debug macro, to be able to turn it
    # off if desired.
    timer :: TimerOutput

    # TODO
    # - add constructor with Cartesian MPI communicator, in case the user
    #   already created one
    # - allow more control on the decomposition directions
    function PencilFFTPlan(size_global::Dims{N},
                           transforms::AbstractTransformList{N},
                           proc_dims::Dims{M}, comm::MPI.Comm,
                           ::Type{T}=Float64;
                           fftw_flags=FFTW.ESTIMATE,
                           fftw_timelimit=FFTW.NO_TIMELIMIT,
                           permute_dims::ValBool=Val(true),
                           transpose_method::AbstractTransposeMethod=
                               TransposeMethods.IsendIrecv(),
                           timer::TimerOutput=TimerOutput(),
                           ibuf=UInt8[], obuf=UInt8[],  # temporary data buffers
                          ) where {N, M, T <: FFTReal}
        g = GlobalFFTParams(size_global, transforms, T)
        t = MPITopology(comm, proc_dims)

        fftw_kw = (:flags => fftw_flags, :timelimit => fftw_timelimit)

        # Options for creation of 1D plans.
        plan1d_opt = (permute_dims=permute_dims,
                      ibuf=ibuf,
                      obuf=obuf,
                      timer=timer,
                      fftw_kw=fftw_kw,
                     )

        plans = _create_plans(g, t, plan1d_opt)
        scale = prod(p -> p.scale_factor, plans)

        new{T, N, M, typeof(g), typeof(plans), typeof(transpose_method)}(
            g, t, plans, scale, transpose_method, ibuf, obuf, timer)
    end

    function PencilFFTPlan(size_global::Dims{N},
                           transform::AbstractTransform,
                           args...; kwargs...) where N
        PencilFFTPlan(size_global, expand_dims(transform, Val(N)),
                      args...; kwargs...)
    end
end

function _create_plans(g::GlobalFFTParams{T, N} where T,
                       topology::MPITopology{M},
                       plan1d_opt::NamedTuple) where {N, M}
    Tin = input_data_type(g)
    transforms = g.transforms
    _create_plans(Tin, g, topology, plan1d_opt, nothing, transforms...)
end

# Create 1D plans recursively.
function _create_plans(::Type{Ti},
                       g::GlobalFFTParams{T, N} where T,
                       topology::MPITopology{M},
                       plan1d_opt::NamedTuple,
                       plan_prev,
                       transform_fw::AbstractTransform,
                       transforms_next::Vararg{AbstractTransform, Ntr}
                      ) where {Ti, N, M, Ntr}
    n = N - Ntr  # current dimension index
    si = g.size_global_in
    so = g.size_global_out

    permute_dims = plan1d_opt.permute_dims::Val === Val(true)
    timer = plan1d_opt.timer

    Pi = if plan_prev === nothing
        # This is the case of the first pencil pair.
        @assert n == 1

        # Generate initial pencils for the first dimension.
        # - Decompose along dimensions "far" from the first one.
        #   Example: if N = 5 and M = 2, then decomp_dims = (4, 5).
        # - No permutation is applied for input data: arrays are accessed in the
        #   natural order (i1, i2, ..., iN).
        decomp_dims = ntuple(m -> N - M + m, Val(M))
        Pencil(topology, si, decomp_dims, Ti, permute=nothing,
               timer=timer)

    else
        Po_prev = plan_prev.pencil_out

        # (i) Determine permutation of pencil data.
        perm = if !permute_dims
            nothing
        elseif Ntr == 0
            # This is the last transform, and I want the index order to be
            # exactly reversed (easier to work with than the alternative below).
            ntuple(i -> N - i + 1, Val(N))  # (N, N-1, ..., 2, 1)
        else
            # Here the data is permuted so that the n-th logical dimension is
            # the first (fastest) dimension in the arrays.
            # The chosen permutation is equivalent to (n, (1:n-1)..., (n+1:N)...).
            t = ntuple(i -> (i == 1) ? n : (i ≤ n) ? (i - 1) : i, Val(N))
            @assert isperm(t)
            @assert t == (n, (1:n-1)..., (n+1:N)...)
            t
        end :: Pencils.OptionalPermutation{N}

        # (ii) Determine decomposed dimensions from the previous
        # decomposition `n - 1`.
        # If `n` was decomposed previously, shift its associated value
        # in `decomp_prev` to the left.
        # Example: if n = 3 and decomp_prev = (1, 3), then decomp = (1, 2).
        decomp_prev = get_decomposition(Po_prev)
        decomp = ntuple(Val(M)) do i
            p = decomp_prev[i]
            p == n ? p - 1 : p
        end

        # Note that if `n` was not decomposed previously, then the
        # decomposed dimensions stay the same.
        @assert n ∈ decomp_prev || decomp === decomp_prev

        # If everything is done correctly, there should be no repeated
        # decomposition dimensions.
        @assert allunique(decomp)

        # Create new pencil sharing some information with Po_prev.
        # (Including data type and dimensions, MPI topology and data buffers.)
        Pencil(Po_prev, decomp_dims=decomp, permute=perm, timer=timer)
    end

    # Output transform along dimension `n`.
    To = eltype_output(transform_fw, eltype(Pi))
    Po = let dims = ntuple(j -> j ≤ n ? so[j] : si[j], Val(N))
        if dims === size_global(Pi) && To === eltype(Pi)
            Pi  # in this case Pi and Po are the same
        else
            Pencil(Pi, To, size_global=dims, timer=timer)
        end
    end

    local scale_bw

    fftplans = let p = get_permutation(Pi)
        dims = if p === nothing
            n  # no index permutation
        else
            # Find index of n-th dimension in the permuted array.
            # If we permuted data to have the n-th dimension as the fastest
            # (leftmost) index, then the result should be 1.
            findfirst(p .== n) :: Int
        end

        # Either there's no permutation and we're transforming the first
        # dimension, or there's a permutation that puts the transformed
        # dimension in the first index.
        @assert !permute_dims || (p === nothing ? n == 1 : first(p) == n)

        # Create temporary arrays with the dimensions required for forward and
        # backward transforms.
        transform_bw = binv(transform_fw)
        pair_fw = transform_fw => _temporary_pencil_array(Pi, plan1d_opt.ibuf)
        pair_bw = transform_bw => _temporary_pencil_array(Po, plan1d_opt.obuf)
        pairs = (pair_fw, pair_bw)

        # Scale factor to be applied after backward transform.
        scale_bw = let tr = pair_bw.first
            # `A` must have the dimensions of the backward transform output
            # (i.e. the forward transform input)
            A = pair_fw.second
            scale_factor(tr, A, dims)
        end

        # Generate forward and backward FFTW transforms.
        fftw_kw = plan1d_opt.fftw_kw
        map(p -> plan(p.first, data(p.second), dims; fftw_kw...), pairs)
    end

    plan_n = PencilPlan1D(Pi, Po, transform_fw, fftplans..., scale_bw)

    (plan_n, _create_plans(To, g, topology, plan1d_opt, plan_n,
                           transforms_next...)...)
end

# No transforms left!
_create_plans(::Type, ::GlobalFFTParams, ::MPITopology, ::NamedTuple,
              plan_prev) = ()

function Base.show(io::IO, p::PencilFFTPlan)
    show(io, p.global_params)
    show(io, p.topology)
    nothing
end

"""
    get_comm(p::PencilFFTPlan)

Get MPI communicator associated to a `PencilFFTPlan`.
"""
get_comm(p::PencilFFTPlan) = get_comm(p.topology)

"""
    get_timer(p::PencilFFTPlan)

Get `TimerOutput` attached to a `PencilFFTPlan`.

See [Measuring performance](@ref PencilFFTs.measuring_performance) for details.
"""
get_timer(p::PencilFFTPlan) = p.timer

# TODO add `destroyable` option? -> create PencilArray from temporary buffer
"""
    allocate_input(p::PencilFFTPlan)

Allocate uninitialised distributed array that can hold input data for the given
plan.
"""
allocate_input(p::PencilFFTPlan) = PencilArray(first(p.plans).pencil_in)

"""
    allocate_output(p::PencilFFTPlan)

Allocate uninitialised distributed array that can hold output data for the
given plan.
"""
allocate_output(p::PencilFFTPlan) = PencilArray(last(p.plans).pencil_out)
