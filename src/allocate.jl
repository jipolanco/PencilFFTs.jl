"""
    allocate_input(p::PencilFFTPlan)          -> PencilArray
    allocate_input(p::PencilFFTPlan, dims...) -> Array{PencilArray}
    allocate_input(p::PencilFFTPlan, Val(N))  -> NTuple{N, PencilArray}

Allocate uninitialised
[`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray)
that can hold input data for the given plan.

The second and third forms respectively allocate an array of `PencilArray`s of
size `dims`, and a tuple of `N` `PencilArray`s.

!!! note "In-place plans"

    If `p` is an in-place real-to-real or complex-to-complex plan, a
    [`ManyPencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.ManyPencilArray)
    is allocated. If `p` is an in-place real-to-complex plan, a
    [`ManyPencilArrayRFFT!`](@ref) is allocated. 
    
    These types hold `PencilArray` wrappers for the input and output transforms (as
    well as for intermediate transforms) which share the same space in memory.
    The input and output `PencilArray`s should be respectively accessed by
    calling [`first(::ManyPencilArray)`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#Base.first-Tuple{ManyPencilArray}) and
    [`last(::ManyPencilArray)`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#Base.last-Tuple{ManyPencilArray}).

    #### Example

    Suppose `p` is an in-place `PencilFFTPlan`. Then,
    ```julia
    @assert is_inplace(p)
    A = allocate_input(p) :: ManyPencilArray
    v_in = first(A)       :: PencilArray  # input data view
    v_out = last(A)       :: PencilArray  # output data view
    ```

    Also note that in-place plans must be performed directly on the returned
    `ManyPencilArray`, and not on the contained `PencilArray` views:
    ```julia
    p * A       # perform forward transform in-place
    p \\ A       # perform backward transform in-place
    # p * v_in  # not allowed!!
    ```
"""
function allocate_input(p::PencilFFTPlan)
    inplace = is_inplace(p)
    _allocate_input(Val(inplace), p)
end

# Out-of-place version
function _allocate_input(inplace::Val{false}, p::PencilFFTPlan)
    T = eltype_input(p)
    pen = pencil_input(p)
    PencilArray{T}(undef, pen, p.extra_dims...)
end

# In-place version
function _allocate_input(inplace::Val{true}, p::PencilFFTPlan)
    (; transforms,) = p.global_params
    _allocate_input(inplace, p, transforms...)
end

# In-place: generic case
function _allocate_input(inplace::Val{true}, p::PencilFFTPlan, transforms...)
    pencils = map(pp -> pp.pencil_in, p.plans)

    # Note that for each 1D plan, the input and output pencils are the same.
    # This is because the datatype stays the same for in-place transforms
    # (in-place real-to-complex transforms are not supported!).
    @assert pencils === map(pp -> pp.pencil_out, p.plans)

    T = eltype_input(p)
    ManyPencilArray{T}(undef, pencils...; extra_dims=p.extra_dims)
end

# In-place: specific case of RFFT!
function _allocate_input(
        inplace::Val{true}, p::PencilFFTPlan{T},
        ::Transforms.RFFT!, ::Vararg{Transforms.FFT!},
    ) where {T}
    plans = p.plans
    pencils = (first(plans).pencil_in, first(plans).pencil_out,  map(pp -> pp.pencil_in, plans[2:end])...)
    ManyPencilArrayRFFT!{T}(undef, pencils...; extra_dims=p.extra_dims)
end

allocate_input(p::PencilFFTPlan, dims...) =
    _allocate_many(allocate_input, p, dims...)

"""
    allocate_output(p::PencilFFTPlan)          -> PencilArray
    allocate_output(p::PencilFFTPlan, dims...) -> Array{PencilArray}
    allocate_output(p::PencilFFTPlan, Val(N))  -> NTuple{N, PencilArray}

Allocate uninitialised [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) that can hold output data for the
given plan.

If `p` is an in-place plan, a [`ManyPencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.ManyPencilArray) is allocated.

See [`allocate_input`](@ref) for details.
"""
function allocate_output(p::PencilFFTPlan)
    inplace = is_inplace(p)
    _allocate_output(Val(inplace), p)
end

# Out-of-place version.
function _allocate_output(inplace::Val{false}, p::PencilFFTPlan)
    T = eltype_output(p)
    pen = pencil_output(p)
    PencilArray{T}(undef, pen, p.extra_dims...)
end

# For in-place plans, the output and input are the same ManyPencilArray.
_allocate_output(inplace::Val{true}, p::PencilFFTPlan) = _allocate_input(inplace, p)

allocate_output(p::PencilFFTPlan, dims...) =
    _allocate_many(allocate_output, p, dims...)

_allocate_many(allocator::Function, p::PencilFFTPlan, dims::Vararg{Int}) =
    [allocator(p) for I in CartesianIndices(dims)]

_allocate_many(allocator::Function, p::PencilFFTPlan, ::Val{N}) where {N} =
    ntuple(n -> allocator(p), Val(N))
