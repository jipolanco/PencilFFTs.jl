"""
    ManyPencilArray{M,T}

Container holding `M` different [`PencilArray`](@ref) views to the same
underlying data buffer of type `T`.

This can be useful to perform in-place operations on `PencilArray` data.

---

    ManyPencilArray(pencils...)

Create a `ManyPencilArray` container that can hold data associated to all the
given [`Pencil`](@ref)s.
"""
struct ManyPencilArray{M, T,
                       Arrays <: Tuple{Vararg{PencilArray,M}},
                      }
    data :: Vector{T}
    arrays :: Arrays

    function ManyPencilArray(
            pencils::Vararg{Pencil{N,X,T}, M} where {N,X}) where {M,T}
        # TODO support extra_dims
        data_length = max(length.(pencils)...)
        data = Vector{T}(undef, data_length)
        arrays = _make_arrays(data, pencils...)
        A = typeof(arrays)
        new{M,T,A}(data, arrays)
    end
end

function _make_arrays(data::Vector, p::Pencil, pens::Vararg{Pencil})
    dims = size_local(p, permute=true)
    n = prod(dims)
    @assert n == length(p)
    vec = view(data, Base.OneTo(n))
    arr = reshape(vec, dims)
    A = PencilArray(p, arr)
    (A, _make_arrays(data, pens...)...)
end

_make_arrays(::Vector) = ()

"""
    length(A::ManyPencilArray)

Returns the number of [`PencilArray`](@ref)s wrapped by `A`.
"""
Base.length(A::ManyPencilArray{M}) where {M} = M

"""
    first(A::ManyPencilArray)

Returns the first [`PencilArray`](@ref) wrapped by `A`.
"""
Base.first(A::ManyPencilArray) = first(A.arrays)

"""
    last(A::ManyPencilArray)

Returns the last [`PencilArray`](@ref) wrapped by `A`.
"""
Base.last(A::ManyPencilArray) = last(A.arrays)

"""
    getindex(A::ManyPencilArray, ::Val{i})
    getindex(A::ManyPencilArray, i::Integer)

Returns the i-th [`PencilArray`](@ref) wrapped by `A`.

If possible, the `Val{i}` form should be prefered, as it is more efficient and
it allows the compiler to know the return type.

See also [`first(::ManyPencilArray)`](@ref), [`last(::ManyPencilArray)`](@ref).

# Example

```julia
A = ManyPencilArray(pencil1, pencil2, pencil3)

# Get the PencilArray associated to `pencil2`.
# u2 = A[2]
u2 = A[Val(2)]  # faster!
```
"""
Base.getindex(A::ManyPencilArray, ::Val{i}) where {i} =
    _getindex(Val(i), A.arrays...)
Base.getindex(A::ManyPencilArray, i) = A[Val(i)]

@inline function _getindex(::Val{i}, a, t::Vararg) where {i}
    i :: Integer
    i <= 0 && throw(BoundsError("index must be >= 1"))
    i === 1 && return a
    _getindex(Val(i - 1), t...)
end

# This will happen if the index `i` intially passed is too large.
@inline _getindex(::Val) = throw(BoundsError("invalid index"))
