## Real-to-real transforms (requires FFTW.jl)

import FFTW: kind2string

const R2R_SUPPORTED_KINDS = (
    FFTW.DHT,
    FFTW.REDFT00,
    FFTW.REDFT01,
    FFTW.REDFT10,
    FFTW.REDFT11,
    FFTW.RODFT00,
    FFTW.RODFT01,
    FFTW.RODFT10,
    FFTW.RODFT11,
)

"""
    R2R{kind}()

Real-to-real transform of type `kind`.

The possible values of `kind` are those described in the
[`FFTW.r2r`](https://juliamath.github.io/FFTW.jl/stable/fft.html#FFTW.r2r)
docs and the [`FFTW`](http://www.fftw.org/doc/) manual:

- [discrete cosine transforms](http://www.fftw.org/doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029):
  `FFTW.REDFT00`, `FFTW.REDFT01`, `FFTW.REDFFT10`, `FFTW.REDFFT11`

- [discrete sine transforms](http://www.fftw.org/doc/1d-Real_002dodd-DFTs-_0028DSTs_0029.html#g_t1d-Real_002dodd-DFTs-_0028DSTs_0029):
  `FFTW.RODFT00`, `FFTW.RODFT01`, `FFTW.RODFFT10`, `FFTW.RODFFT11`

- [discrete Hartley transform](http://www.fftw.org/doc/1d-Discrete-Hartley-Transforms-_0028DHTs_0029.html#g_t1d-Discrete-Hartley-Transforms-_0028DHTs_0029):
  `FFTW.DHT`

Note: [half-complex format
DFTs](http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html#The-Halfcomplex_002dformat-DFT)
(`FFTW.R2HC`, `FFTW.HC2R`) are not currently supported.

"""
struct R2R{kind} <: AbstractTransform
    function R2R{kind}() where kind
        if kind ∉ R2R_SUPPORTED_KINDS
            throw(ArgumentError(
                "unsupported r2r transform kind: $(kind2string(kind))"))
        end
        new()
    end
end

"""
    R2R!{kind}()

In-place version of [`R2R`](@ref).

See also [`FFTW.r2r!`](https://juliamath.github.io/FFTW.jl/stable/fft.html#FFTW.r2r!).
"""
struct R2R!{kind} <: AbstractTransform
    function R2R!{kind}() where kind
        R2R{kind}()  # executes verification code above
        new()
    end
end

const AnyR2R{kind} = Union{R2R{kind}, R2R!{kind}} where {kind}

is_inplace(::R2R) = false
is_inplace(::R2R!) = true

Base.show(io::IO, tr::R2R) = print(io, "R2R{", kind2string(kind(tr)), "}")
Base.show(io::IO, tr::R2R!) = print(io, "R2R!{", kind2string(kind(tr)), "}")

"""
    kind(transform::R2R)

Get `kind` of real-to-real transform.
"""
kind(::AnyR2R{K}) where {K} = K

length_output(::AnyR2R, length_in::Integer) = length_in
eltype_input(::AnyR2R, ::Type{T}) where {T} = T
eltype_output(::AnyR2R, ::Type{T}) where {T} = T

# NOTE: plan_r2r is type-unstable!!
# More precisely, plan_r2r returns a FFTW.r2rFFTWPlan{T, K, inplace, N},
# whose second parameter `K` is "a tuple of the transform kinds along each
# dimension". That is, `K` is a tuple of the form `(kind1, kind2, ..., kindN)`
# with the same length as `dims`.
#
# Since we have static information on the kind (it's a parameter of the R2R
# type), we try to work around the issue by typing the return type. This
# function will be type-stable if `dims` has a static length, i.e. if
# `length(dims)` is known by the compiler. This will be the case if `dims` is a
# tuple or a scalar value (e.g. `(1, 3)` or `1`), but not if it is a range (e.g.
# `2:3`).
function plan(transform::AnyR2R, A, dims; kwargs...)
    kd = kind(transform)
    K = ntuple(_ -> kd, length(dims))
    R = FFTW.r2rFFTWPlan{T,K} where {T}  # try to guess the return type
    _plan_r2r(transform, A, kd, dims; kwargs...) :: R
end

_plan_r2r(::R2R, args...; kwargs...) = FFTW.plan_r2r(args...; kwargs...)
_plan_r2r(::R2R!, args...; kwargs...) = FFTW.plan_r2r!(args...; kwargs...)

# Scale factors for r2r transforms.
scale_factor(::AnyR2R, A, dims) = _prod_dims(2 .* size(A), dims)
scale_factor(::AnyR2R{FFTW.REDFT00}, A, dims) =
    _prod_dims(2 .* (size(A) .- 1), dims)
scale_factor(::AnyR2R{FFTW.RODFT00}, A, dims) =
    _prod_dims(2 .* (size(A) .+ 1), dims)
scale_factor(::AnyR2R{FFTW.DHT}, A, dims) = _prod_dims(A, dims)

for T in (:R2R, :R2R!)
    @eval begin
        # From FFTW docs (4.8.3 1d Real-even DFTs (DCTs)):
        #   The unnormalized inverse of REDFT00 is REDFT00, of REDFT10 is REDFT01 and
        #   vice versa, and of REDFT11 is REDFT11.
        #   Each unnormalized inverse results in the original array multiplied by N,
        #   where N is the logical DFT size. For REDFT00, N=2(n-1) (note that n=1 is not
        #   defined); otherwise, N=2n.
        binv(::$T{FFTW.REDFT00}) = $T{FFTW.REDFT00}()
        binv(::$T{FFTW.REDFT01}) = $T{FFTW.REDFT10}()
        binv(::$T{FFTW.REDFT10}) = $T{FFTW.REDFT01}()
        binv(::$T{FFTW.REDFT11}) = $T{FFTW.REDFT11}()

        # From FFTW docs (4.8.4 1d Real-odd DFTs (DSTs)):
        #    The unnormalized inverse of RODFT00 is RODFT00, of RODFT10 is RODFT01 and
        #    vice versa, and of RODFT11 is RODFT11.
        #    Each unnormalized inverse results in the original array multiplied by N,
        #    where N is the logical DFT size. For RODFT00, N=2(n+1); otherwise, N=2n.
        binv(::$T{FFTW.RODFT00}) = $T{FFTW.RODFT00}()
        binv(::$T{FFTW.RODFT01}) = $T{FFTW.RODFT10}()
        binv(::$T{FFTW.RODFT10}) = $T{FFTW.RODFT01}()
        binv(::$T{FFTW.RODFT11}) = $T{FFTW.RODFT11}()

        # From FFTW docs (4.8.5 1d Discrete Hartley Transforms (DHTs)):
        #    [...] applying the transform twice (the DHT is its own inverse) will
        #    multiply the input by n.
        binv(::$T{FFTW.DHT}) = $T{FFTW.DHT}()
    end
end

expand_dims(::F, ::Val{N}) where {F <: AnyR2R, N} =
    N === 0 ? () : (F(), expand_dims(F(), Val(N - 1))...)
