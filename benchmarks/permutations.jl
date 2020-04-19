#!/usr/bin/env julia

using PencilFFTs.PencilArrays
using BenchmarkTools

abstract type IterationOrder end
struct OrderSrc <: IterationOrder end
struct OrderDst <: IterationOrder end

iter_indices(::OrderSrc, src, dst) = CartesianIndices(src)
iter_indices(::OrderDst, src, dst) = CartesianIndices(dst)

indices(::OrderSrc, I, perm) = (I, PencilArrays.permute_indices(I, perm))
indices(::OrderDst, J, perm) = let iperm = PencilArrays.inverse_permutation(perm)
    (PencilArrays.permute_indices(J, iperm), J)
end

function copy_permuted!(dst::AbstractArray{To,3}, src::AbstractArray{Ti,3},
                        perm, order::IterationOrder) where {To,Ti}
    @assert length(src) == length(dst)

    for C in iter_indices(order, src, dst)
        I, J = indices(order, C, perm)
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

    print("  copy_permuted (src order)...")
    @btime copy_permuted!($dst, $src, $pval, OrderSrc())

    print("  copy_permuted (dst order)...")
    @btime copy_permuted!($dst, $src, $pval, OrderDst())

    if perm === (1, 2, 3)
        print("  copyto...                   ")
        @btime copyto!($dst, $src)
    end

    print("  permutedims...              ")
    @btime permutedims!($dst, $src, $perm)

    print("  PermutedDimsArray(src)...   ")
    src_p = PermutedDimsArray{T,N,perm,iperm,typeof(src)}(src)
    @btime copyto!($dst, $src_p)  # uses generic copyto!

    print("  PermutedDimsArray(dst)...   ")
    dst_p = PermutedDimsArray{T,N,iperm,perm,typeof(dst)}(dst)
    @btime copyto!($dst_p, $src)  # copyto! for PermutedDimsArray (generally faster)

    println()

    nothing
end

main()
