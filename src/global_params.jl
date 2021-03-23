"""
    GlobalFFTParams{T, N, inplace}

Specifies the global parameters for an N-dimensional distributed transform.
These include the element type `T` and global data sizes of input and output
data, as well as the transform types to be performed along each dimension.

---

    GlobalFFTParams(size_global, transforms, [real_type=Float64])

Define parameters for N-dimensional transform.

`transforms` must be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[`Transform types`](@ref Transforms).

The element type must be a real type accepted by FFTW, i.e. either `Float32` or
`Float64`.

Note that the transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms.

# Example

To perform a 3D FFT of real data, first a real-to-complex FFT must be applied
along the first dimension, followed by two complex-to-complex FFTs along the
other dimensions:

```jldoctest
julia> size_global = (64, 32, 128);  # size of real input data

julia> transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT());

julia> fft_params = PencilFFTs.GlobalFFTParams(size_global, transforms)
Transforms: (RFFT, FFT, FFT)
Input type: Float64
Global dimensions: (64, 32, 128) -> (33, 32, 128)
```

"""
struct GlobalFFTParams{T, N, inplace, F <: AbstractTransformList{N}}
    # Transforms to be applied along each dimension.
    transforms :: F

    size_global_in  :: Dims{N}
    size_global_out :: Dims{N}

    function GlobalFFTParams(size_global::Dims{N},
                             transforms::AbstractTransformList{N},
                             ::Type{T}=Float64,
                            ) where {N, T <: FFTReal}
        F = typeof(transforms)
        size_global_out = length_output.(transforms, size_global)
        inplace = is_inplace(transforms...)
        if inplace === nothing
            throw(ArgumentError(
                "cannot combine in-place and out-of-place transforms: $(transforms)"))
        end
        new{T, N, inplace, F}(transforms, size_global, size_global_out)
    end
end

Base.ndims(::Type{<:GlobalFFTParams{T,N}}) where {T,N} = N
Base.ndims(g::GlobalFFTParams) = ndims(typeof(g))
Transforms.is_inplace(g::GlobalFFTParams{T,N,I}) where {T,N,I} = I

function Base.show(io::IO, g::GlobalFFTParams)
    print(io, "Transforms: ", g.transforms)
    print(io, "\nInput type: ", input_data_type(g))
    print(io, "\nGlobal dimensions: ",
          g.size_global_in, " -> ", g.size_global_out)
    nothing
end

# Determine input data type for multidimensional transform.
input_data_type(g::GlobalFFTParams{T}) where {T} =
    _input_data_type(T, g.transforms...) :: DataType

function _input_data_type(
        ::Type{T}, transform::AbstractTransform, etc...,
    ) where {T}
    Tin = eltype_input(transform, T) :: DataType
    if isnothing(Tin)
        # This is the case if `transform` can take both real and complex data.
        # We check the next transform type.
        return _input_data_type(T, etc...)
    end
    Tin
end

# If all calls to `eltype_input` return Nothing, then we return the given real
# type. This will be the case for combinations of real-to-real transforms.
_input_data_type(::Type{T}) where {T} = T
