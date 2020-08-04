module Permutations

export Permutation, NoPermutation

export
    is_valid_permutation,
    check_permutation,
    permute_indices,
    relative_permutation,
    inverse_permutation,
    identity_permutation,
    is_identity_permutation,
    append_to_permutation


include("types.jl")
include("operations.jl")

end
