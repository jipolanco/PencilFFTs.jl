"""
    GlobalFFTParams{N}

Specifies the global parameters for an N-dimensional distributed transform.
These include the global data sizes of input and output data, as well as the
transform types to be performed along each dimension.

---

    GlobalFFTParams(size_global_in::Dims{N},
                    transform_types::F) where {N, F<:Tuple}

Define parameters for N-dimensional transform.

`transform_types` must be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
`Transforms.AbstractTransform`. For all the possible transforms, see
[`Transform types`](@ref Transforms).

Note that the transforms are applied one dimension at a time, with the leftmost
dimension first.

# Example

To perform a 3D FFT of real data, first a real-to-complex FFT must be applied
along the first dimension, followed by two complex-to-complex FFTs along the
other dimensions:

```julia
size_global_in = (64, 32, 128)  # size of real input data
transforms = (Transform.RFFT, Transform.FFT, Transform.FFT)
fft_params = GlobalFFTParams(size_global_in, transforms)
```

"""
struct GlobalFFTParams{N, F}
    size_global_in :: Dims{N}

    # Transforms to be applied in each direction.
    # F = Tuple{F_1, F_2, ..., F_N}, where F_n <: AbstractTransform.
    transform_types :: F

    function GlobalFFTParams(size_global_in::Dims{N},
                             transform_types::F) where {N, F<:Tuple}
        @assert length(transform_types) == N
        @assert eltype(transform_types) <: Transforms.AbstractTransform
        new{N, F}(size_global_in, transform_types)
    end
end
