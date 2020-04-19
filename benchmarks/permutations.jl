#!/usr/bin/env julia

using PencilFFTs.PencilArrays
using BenchmarkTools

function copy_permuted!(dst::AbstractArray{To,3}, src::AbstractArray{Ti,3},
                        perm) where {To,Ti}
    @assert length(src) == length(dst)

    # To iterate in destination order:
    #   iperm = PencilArrays.inverse_permutation(perm)
    #   for J in CartesianIndices(dst)
    #       I = PencilArrays.permute_indices(J, iperm)

    for I in CartesianIndices(src)
        J = PencilArrays.permute_indices(I, perm)
        @inbounds dst[J] = src[I]
    end

    dst
end

function main()
    N = 64
    N2 = N >> 1
    N4 = N >> 2
    Nxyz = (N, N, N)

    dst = zeros(Nxyz...)
    dst_range = (1:N, 1:N2, 1:N4)
    dst_view_dims = length.(dst_range)
    dst_view = view(dst, dst_range...)

    bench_permutation(dst_view, Val((1, 2, 3)))
    bench_permutation(dst_view, Val((1, 3, 2)))

    bench_permutation(dst_view, Val((2, 1, 3)))
    bench_permutation(dst_view, Val((2, 3, 1)))

    bench_permutation(dst_view, Val((3, 1, 2)))
    bench_permutation(dst_view, Val((3, 2, 1)))
end

function bench_permutation(dst::AbstractArray{T,N},
                           pval::Val{perm}) where {T,N,perm}
    iperm = invperm(perm)
    Ndst = length(dst)
    src_vec = zeros(T, 2Ndst)
    src_dims = ntuple(d -> size(dst, iperm[d]), Val(N))
    src = reshape(view(src_vec, 1:Ndst), src_dims)

    println("Permutation: $perm, size(dst) = ", size(dst))

    print("  copy_permuted...          ")
    @btime copy_permuted!($dst, $src, $pval)
    if perm === (1, 2, 3)
        print("  copy_permuted (nothing)...")
        @btime copy_permuted!($dst, $src, nothing)
        print("  copyto...                 ")
        @btime copyto!($dst, $src)
    end

    print("  permutedims...            ")
    @btime permutedims!($dst, $src, $perm)

    print("  PermutedDimsArray(src)... ")
    src_p = PermutedDimsArray{T,N,perm,iperm,typeof(src)}(src)
    @btime copyto!($dst, $src_p)  # uses generic copyto!

    print("  PermutedDimsArray(dst)... ")
    dst_p = PermutedDimsArray{T,N,iperm,perm,typeof(dst)}(dst)
    @btime copyto!($dst_p, $src)  # copyto! for PermutedDimsArray (generally faster)

    println()

    nothing
end

main()
