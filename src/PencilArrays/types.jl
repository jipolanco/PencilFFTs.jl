"""
    Permutation{p}

Describes a compile-time dimension permutation.

The type parameter `p` should be a valid permutation such as `(3, 1, 2)`.

The parameter `p` may also be `nothing`.
A `Permutation{nothing}` represents an identity permutation: it is equivalent to
`Permutation{(1, 2, …, N)}` for any number of dimensions `N`.
The [`NoPermutation`](@ref) type is provided as an alias for
`Permutation{nothing}`.

---

    Permutation(perm::Vararg{Int})
    Permutation(perm::NTuple{N,Int})

Constructs a `Permutation`.

# Example

Both are equivalent:

```julia
p1 = Permutation(3, 4)
p2 = Permutation((3, 4))
```

---

    Permutation(nothing)
    NoPermutation()

Constructs an identity permutation.
"""
struct Permutation{p, N}  # TODO do we need the N parameter?
    Permutation{nothing, 0}() = new{nothing, 0}()
    Permutation(perm::Vararg{Int}) = new{perm, length(perm)}()
end

Permutation(::Nothing) = Permutation{nothing, 0}()
Permutation(perm::Tuple) = Permutation(perm...)

"""
    NoPermutation

Alias for an identity permutation, i.e. `NoPermutation = Permutation{nothing}`.

This alias can be called as a constructor.
In other words, `NoPermutation()` creates a `Permutation{nothing}` as expected.
"""
const NoPermutation = typeof(Permutation(nothing))

Base.show(io::IO, ::NoPermutation) = print(io, "None")
Base.show(io::IO, ::Permutation{p}) where {p} = print(io, p)
