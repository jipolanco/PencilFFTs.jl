const AbstractTransformList{N} = NTuple{N, Transforms.AbstractTransform} where N

"""
    GlobalFFTParams{N}

Specifies the global parameters for an N-dimensional distributed transform.
These include the global data sizes of input and output data, as well as the
transform types to be performed along each dimension.

---

    GlobalFFTParams(size_global, transforms)

Define parameters for N-dimensional transform.

`transforms` must be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
[`Transforms.AbstractTransform`](@ref). For all the possible transforms, see
[`Transform types`](@ref Transforms).

Note that the transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms.

# Example

To perform a 3D FFT of real data, first a real-to-complex FFT must be applied
along the first dimension, followed by two complex-to-complex FFTs along the
other dimensions:

```julia
size_global = (64, 32, 128)  # size of real input data
transforms = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())
fft_params = GlobalFFTParams(size_global, transforms)
```

"""
struct GlobalFFTParams{N, F <: AbstractTransformList{N}}
    # Transforms to be applied along each dimension.
    transforms :: F

    size_global_in  :: Dims{N}
    size_global_out :: Dims{N}

    function GlobalFFTParams(size_global::Dims{N},
                             transforms::AbstractTransformList{N}) where {N}
        # TODO
        # - verify that r2c dimensions have even size, as currently required by
        #   the definition of `length_output` (is this really necessary? try to
        #   support odd sizes)
        F = typeof(transforms)
        size_global_out = Transforms.length_output.(transforms, size_global)
        new{N, F}(transforms, size_global, size_global_out)
    end
end
