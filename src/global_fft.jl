const AbstractTransformList{N} = NTuple{N, AbstractTransform} where N

"""
    GlobalFFTParams{T, N}

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

julia> fft_params = PencilFFTs.GlobalFFTParams(size_global, transforms);
```

"""
struct GlobalFFTParams{T, N, F <: AbstractTransformList{N}}
    # Transforms to be applied along each dimension.
    transforms :: F

    size_global_in  :: Dims{N}
    size_global_out :: Dims{N}

    function GlobalFFTParams(size_global::Dims{N},
                             transforms::AbstractTransformList{N},
                             ::Type{T}=Float64,
                            ) where {N, T <: FFTReal}
        # TODO
        # - verify that r2c dimensions have even size, as currently required by
        #   the definition of `length_output` (is this really necessary? try to
        #   support odd sizes)
        F = typeof(transforms)
        size_global_out = length_output.(transforms, size_global)
        new{T, N, F}(transforms, size_global, size_global_out)
    end
end

# Determine input data type for multidimensional transform.
# It will return Nothing if the data type can't be resolved from the transform
# list. This will be the case if `g.transforms` is only made of `NoTransform`s.
input_data_type(g::GlobalFFTParams{T}) where T =
    input_data_type(T, g.transforms...) :: DataType

function input_data_type(::Type{T}, transform::AbstractTransform,
                         next_transforms::Vararg{AbstractTransform}) where T
    Tin = eltype_input(transform, T) :: DataType
    @debug "input_data_type" Tin transform next_transforms
    if Tin === Nothing
        # Check the next transform type.
        @assert transform isa Transforms.NoTransform
        return input_data_type(T, next_transforms...)
    end
    Tin
end

input_data_type(::Type{T}, transform::AbstractTransform) where T =
    eltype_input(transform, T)
