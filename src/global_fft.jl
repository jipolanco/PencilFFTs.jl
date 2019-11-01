const TransformList{N} = NTuple{N, Transforms.AbstractTransform} where N

"""
    GlobalFFTParams{N}

Specifies the global parameters for an N-dimensional distributed transform.
These include the global data sizes of input and output data, as well as the
transform types to be performed along each dimension.

---

    GlobalFFTParams(size_global_in::Dims{N},
                    transforms::NTuple{N, AbstractTransform}) where N

Define parameters for N-dimensional transform.

`transforms` must be a tuple of length `N` specifying the transforms to be
applied along each dimension. Each element must be a subtype of
`Transforms.AbstractTransform`. For all the possible transforms, see
[`Transform types`](@ref Transforms).

Note that the transforms are applied one dimension at a time, with the leftmost
dimension first for forward transforms.

# Example

To perform a 3D FFT of real data, first a real-to-complex FFT must be applied
along the first dimension, followed by two complex-to-complex FFTs along the
other dimensions:

```julia
size_global_in = (64, 32, 128)  # size of real input data
transforms = (Transform.RFFT(), Transform.FFT(), Transform.FFT())
fft_params = GlobalFFTParams(size_global_in, transforms)
```

"""
struct GlobalFFTParams{N, F <: TransformList{N}}
    # Transforms to be applied along each dimension.
    transforms :: F

    size_global_in  :: Dims{N}
    size_global_out :: Dims{N}

    function GlobalFFTParams(size_global_in::Dims{N},
                             transforms::TransformList{N}) where {N}
        # TODO
        # - verify that r2c dimensions have even size, as currently required by
        #   the definition of `length_output` (is this really necessary? try to
        #   support odd sizes)
        F = typeof(transforms)
        size_global_out = Transforms.length_output.(transforms, size_global_in)
        new{N, F}(transforms, size_global_in, size_global_out)
    end
end
