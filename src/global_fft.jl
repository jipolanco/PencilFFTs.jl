"""
    GlobalFFTParams{N}

Specifies the global parameters for an N-dimensional distributed transform.
These include the global data sizes of input and output data, as well as the
transform types to be performed along each dimension.
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
