module Permutations

export Permutation, NoPermutation

# TODO remove unused stuff
export
    extract,
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
