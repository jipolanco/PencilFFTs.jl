# copied and modified from https://github.com/jipolanco/PencilArrays.jl/blob/master/src/multiarrays.jl

"""
    ManyPencilArrayInplaceRFFT{T,N,M}

Container holding `M` different [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) views to the same
underlying data buffer. All views share the same and dimensionality `N`.
The element type `T` of the first view is real, that of subsequent views is 
`Complex{T}`. 

This can be used to perform in-place real-to-complex plan, see also[`Tranforms.RFFT!`](@ref). 
It is used internally for such transforms by [`allocate_input`](@ref) and should not be constructed directly.

---

    ManyPencilArrayInplaceRFFT{T}(undef, pencils...; extra_dims=())

Create a `ManyPencilArrayInplaceRFFT` container that can hold data of type `T` and `Complex{T}` associated
to all the given [`Pencil`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.Pencil)s.

The optional `extra_dims` argument is the same as for [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray).

See also [`ManyPencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.ManyPencilArray)
"""
struct ManyPencilArrayInplaceRFFT{
        T,  # element type of real array
        N,  # number of dimensions of each array (including extra_dims)
        M,  # number of arrays
        Arrays <: Tuple{Vararg{PencilArray,M}},
        DataVector <: AbstractVector{T},
        DataVectorComplex <: AbstractVector{Complex{T}},
    }
    data   :: DataVector
    data_complex   :: DataVectorComplex
    arrays :: Arrays

    function ManyPencilArrayInplaceRFFT{T}(
            init, real_pencil::Pencil{Np}, complex_pencils::Vararg{Pencil{Np}};
            extra_dims::Dims=()
        ) where {Np,T}
        # real_pencil is a Pencil with dimensions `dims` of a real array with no padding and no permutation 
        # the padded dimensions are (2*(dims[1] ÷ 2 + 1), dims[2:end]...)  
        # first(complex_pencils) is a Pencil with dimensions of a complex array (dims[1] ÷ 2 + 1, dims[2:end]...) and no permutation 
        pencils = (real_pencil, complex_pencils...)
        BufType = PencilArrays.typeof_array(real_pencil)
        @assert all(p -> PencilArrays.typeof_array(p) === BufType, complex_pencils)
        # TODO : checks dims of pencils
        data_length = max(2 .* length.(complex_pencils)...) * prod(extra_dims)
        data_real = BufType{T}(init, data_length)
        fill_with_zeros && fill!(data_real, 0.0)
        # we don't use data_complex = reinterpret(Complex{T}, data_real)
        # since there is an issue with StridedView of ReinterpretArray, called by _permutedims in PencilArrays.Transpositions
        ptr_complex = convert(Ptr{Complex{T}}, pointer(data_real)) 
        data_complex = unsafe_wrap(BufType, ptr_complex, data_length ÷ 2)  
        array_real = _make_real_array(data_real, extra_dims, real_pencil)
        arrays_complex = _make_arrays(data_complex, extra_dims, complex_pencils...)
        arrays = (array_real, arrays_complex...)
        N = Np + length(extra_dims)
        M = length(pencils)
        new{T, N, M, typeof(arrays), typeof(data_real), typeof(data_complex)}(data_real, data_complex, arrays)
    end
end

function _make_real_array(data, extra_dims, p)
    dims_space_local = size_local(p, MemoryOrder())
    dims_padded_local = (2*(dims_space_local[1] ÷ 2 + 1), dims_space_local[2:end]...)
    dims = (dims_padded_local..., extra_dims...)
    axes_local = (Base.OneTo.(dims_space_local)..., Base.OneTo.(extra_dims)...)
    n = prod(dims)
    vec = view(data, Base.OneTo(n))
    parent_arr = reshape(vec, dims)
    arr = view(parent_arr, axes_local...)
    PencilArray(p, arr)
end

function _make_arrays(data::V, extra_dims::Dims, p::Pencil,
    pens::Vararg{Pencil}) where {V<:Union{DV, Base.ReinterpretArray{Complex{T}, 1, T, DV}}} where {DV <: DenseVector{T}} where {T}
    dims = (size_local(p, MemoryOrder())..., extra_dims...)
    n = prod(dims)
    @assert n == length_local(p) * prod(extra_dims)
    vec = view(data, Base.OneTo(n))
    arr = reshape(vec, dims)
    A = PencilArray(p, arr)
    (A, _make_arrays(data, extra_dims, pens...)...)
end

_make_arrays(::V, ::Dims) where {V<:Union{DV, Base.ReinterpretArray{Complex{T}, 1, T, DV}}} where {DV <: DenseVector{T}} where {T} = ()

Base.eltype(::ManyPencilArrayInplaceRFFT{T}) where {T} = T
Base.ndims(::ManyPencilArrayInplaceRFFT{T,N}) where {T,N} = N

"""
    Tuple(A::ManyPencilArrayInplaceRFFT) -> (u1, u2, …)

Returns the [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray)s wrapped by `A`.

This can be useful for iterating over all the wrapped arrays.
"""
@inline Base.Tuple(A::ManyPencilArrayInplaceRFFT) = A.arrays

"""
    length(A::ManyPencilArrayInplaceRFFT)

Returns the number of [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray)s wrapped by `A`.
"""
Base.length(::ManyPencilArrayInplaceRFFT{T,N,M}) where {T,N,M} = M

"""
    first(A::ManyPencilArrayInplaceRFFT)

Returns the first [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) wrapped by `A`.
"""
Base.first(A::ManyPencilArrayInplaceRFFT) = first(A.arrays)

"""
    last(A::ManyPencilArrayInplaceRFFT)

Returns the last [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) wrapped by `A`.
"""
Base.last(A::ManyPencilArrayInplaceRFFT) = last(A.arrays)

"""
    getindex(A::ManyPencilArrayInplaceRFFT, ::Val{i})
    getindex(A::ManyPencilArrayInplaceRFFT, i::Integer)

Returns the i-th [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) wrapped by `A`.

If possible, the `Val{i}` form should be preferred, as it ensures that the full
type of the returned `PencilArray` is known by the compiler.

See also [`first(::ManyPencilArrayInplaceRFFT)`](@ref), [`last(::ManyPencilArrayInplaceRFFT)`](@ref).

# Example

```julia
A = ManyPencilArrayInplaceRFFT(pencil1, pencil2, pencil3)

# Get the PencilArray associated to `pencil2`.
u2 = A[2]
u2 = A[Val(2)]
```
"""
Base.getindex(A::ManyPencilArrayInplaceRFFT, ::Val{i}) where {i} =
    _getindex(Val(i), A.arrays...)
@inline Base.getindex(A::ManyPencilArrayInplaceRFFT, i) = A[Val(i)]

@inline function _getindex(::Val{i}, a, t::Vararg) where {i}
    i :: Integer
    i <= 0 && throw(BoundsError("index must be >= 1"))
    i == 1 && return a
    _getindex(Val(i - 1), t...)
end

# This will happen if the index `i` intially passed is too large.
@inline _getindex(::Val) = throw(BoundsError("invalid index"))