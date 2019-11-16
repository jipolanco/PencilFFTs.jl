## Real-to-real transforms (requires FFTW.jl)

# TODO
# - Support DHT and R2HC/HC2R

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

- [halfcomplex-format DFT](http://www.fftw.org/doc/The-Halfcomplex_002dformat-DFT.html#The-Halfcomplex_002dformat-DFT):
  `FFTW.R2HC`, `FFTW.HC2R`

"""
struct R2R{kind} <: AbstractTransform end

# Discrete Cosine transforms.
const DCT = Union{R2R{FFTW.REDFT00}, R2R{FFTW.REDFT01},
                  R2R{FFTW.REDFT10}, R2R{FFTW.REDFT11}}

# Discrete Sine transforms.
const DST = Union{R2R{FFTW.RODFT00}, R2R{FFTW.RODFT01},
                  R2R{FFTW.RODFT10}, R2R{FFTW.RODFT11}}

"""
    kind(transform::R2R)

Get `kind` of real-to-real transform.
"""
kind(::R2R{K}) where K = K

length_output(::R2R, length_in::Integer) = length_in
eltype_input(::R2R, ::Type{T}) where T = T
eltype_output(::R2R, ::Type{T}) where T = T

_kinds(::Val{kind}, ::Val{N}) where {kind, N} = ntuple(_ -> kind, N)

# NOTE: plan_r2r is type-unstable!!
# More precisely, plan_r2r returns a FFTW.r2rFFTWPlan{T, K, inplace, N},
# whose second parameter `K` is "a tuple of the transform kinds along each
# dimension.". That is, `K` is a tuple of the form `(kind, kind, ..., kind)`
# with the same length as `dims`.
#
# Since we have static information on the kind (it's a parameter of the R2R
# type), we try to work around the issue by typing the return type. This
# function will be type-stable if `dims` has a static length, i.e. if
# `length(dims)` is known by the compiler. This will be the case if `dims` is a
# tuple or a scalar value (e.g. `(1, 3)` or `1`), but not if it is a range (e.g.
# `2:3`).
function plan(transform::R2R, A, dims; kwargs...)
    kd = kind(transform)
    K = _kinds(Val(kd), Val(length(dims)))
    R = FFTW.r2rFFTWPlan{T, K} where T  # try to guess the return type
    FFTW.plan_r2r(A, kd, dims; kwargs...) :: R
end

# From FFTW docs (4.8.3 1d Real-even DFTs (DCTs)):
#   The unnormalized inverse of REDFT00 is REDFT00, of REDFT10 is REDFT01 and
#   vice versa, and of REDFT11 is REDFT11.
#   Each unnormalized inverse results in the original array multiplied by N,
#   where N is the logical DFT size. For REDFT00, N=2(n-1) (note that n=1 is not
#   defined); otherwise, N=2n.
binv(::R2R{FFTW.REDFT00}) = R2R{FFTW.REDFT00}()
binv(::R2R{FFTW.REDFT01}) = R2R{FFTW.REDFT10}()
binv(::R2R{FFTW.REDFT10}) = R2R{FFTW.REDFT01}()
binv(::R2R{FFTW.REDFT11}) = R2R{FFTW.REDFT11}()

scale_factor(::DCT, A, dims) =
    _prod_dims(2 .* size(A), dims)
scale_factor(::R2R{FFTW.REDFT00}, A, dims) =
    _prod_dims(2 .* (size(A) .- 1), dims)

# From FFTW docs (4.8.4 1d Real-odd DFTs (DSTs)):
#    The unnormalized inverse of RODFT00 is RODFT00, of RODFT10 is RODFT01 and
#    vice versa, and of RODFT11 is RODFT11.
#    Each unnormalized inverse results in the original array multiplied by N,
#    where N is the logical DFT size. For RODFT00, N=2(n+1); otherwise, N=2n. 
binv(::R2R{FFTW.RODFT00}) = R2R{FFTW.RODFT00}()
binv(::R2R{FFTW.RODFT01}) = R2R{FFTW.RODFT10}()
binv(::R2R{FFTW.RODFT10}) = R2R{FFTW.RODFT01}()
binv(::R2R{FFTW.RODFT11}) = R2R{FFTW.RODFT11}()

scale_factor(::DST, A, dims) =
    _prod_dims(2 .* size(A), dims)
scale_factor(::R2R{FFTW.RODFT00}, A, dims) =
    _prod_dims(2 .* (size(A) .+ 1), dims)

expand_dims(::F, ::Val{N}) where {F <: R2R, N} =
    N === 0 ? () : (F(), expand_dims(F(), Val(N - 1))...)
