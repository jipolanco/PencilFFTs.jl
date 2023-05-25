# copied and modified from https://github.com/jipolanco/PencilArrays.jl/blob/master/src/multiarrays.jl
import PencilArrays: AbstractManyPencilArray, _make_arrays

"""
    ManyPencilArrayRFFT!{T,N,M} <: AbstractManyPencilArray{N,M}

Container holding `M` different [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray) views to the same
underlying data buffer. All views share the same and dimensionality `N`.
The element type `T` of the first view is real, that of subsequent views is 
`Complex{T}`. 

This can be used to perform in-place real-to-complex plan, see also[`Transforms.RFFT!`](@ref). 
It is used internally for such transforms by [`allocate_input`](@ref) and should not be constructed directly.

---

    ManyPencilArrayRFFT!{T}(undef, pencils...; extra_dims=())

Create a `ManyPencilArrayRFFT!` container that can hold data of type `T` and `Complex{T}` associated
to all the given [`Pencil`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.Pencil)s.

The optional `extra_dims` argument is the same as for [`PencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.PencilArray).

See also [`ManyPencilArray`](https://jipolanco.github.io/PencilArrays.jl/dev/PencilArrays/#PencilArrays.ManyPencilArray)
"""
struct ManyPencilArrayRFFT!{
        T,  # element type of real array
        N,  # number of dimensions of each array (including extra_dims)
        M,  # number of arrays
        Arrays <: Tuple{Vararg{PencilArray,M}},
        DataVector <: AbstractVector{T},
        DataVectorComplex <: AbstractVector{Complex{T}},
    } <: AbstractManyPencilArray{N, M}
    data   :: DataVector
    data_complex   :: DataVectorComplex
    arrays :: Arrays

    function ManyPencilArrayRFFT!{T}(
            init, real_pencil::Pencil{Np}, complex_pencils::Vararg{Pencil{Np}};
            extra_dims::Dims=()
        ) where {Np,T<:FFTReal}
        # real_pencil is a Pencil with dimensions `dims` of a real array with no padding and no permutation 
        # the padded dimensions are (2*(dims[1] ÷ 2 + 1), dims[2:end]...)  
        # first(complex_pencils) is a Pencil with dimensions of a complex array (dims[1] ÷ 2 + 1, dims[2:end]...) and no permutation 
        pencils = (real_pencil, complex_pencils...)
        BufType = PencilArrays.typeof_array(real_pencil)
        @assert all(p -> PencilArrays.typeof_array(p) === BufType, complex_pencils)
        @assert size_global(real_pencil)[2:end] ==  size_global(first(complex_pencils))[2:end] 
        @assert first(size_global(real_pencil)) ÷ 2 + 1 ==  first(size_global(first(complex_pencils)))

        data_length = max(2 .* length.(complex_pencils)...) * prod(extra_dims)
        data_real = BufType{T}(init, data_length)

        # we don't use data_complex = reinterpret(Complex{T}, data_real)
        # since there is an issue with StridedView of ReinterpretArray, called by _permutedims in PencilArrays.Transpositions
        ptr_complex = convert(Ptr{Complex{T}}, pointer(data_real)) 
        data_complex = unsafe_wrap(BufType, ptr_complex, data_length ÷ 2)  
        
        array_real = _make_real_array(data_real, extra_dims, real_pencil)
        arrays_complex = PencilArrays._make_arrays(data_complex, extra_dims, complex_pencils...)
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
