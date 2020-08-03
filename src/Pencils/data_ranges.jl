# Functions determining local data ranges in the different pencil
# configurations.

function local_data_range(p, P, N)
    @assert 1 <= p <= P
    a = (N * (p - 1)) ÷ P + 1
    b = (N * p) ÷ P
    a:b
end

# "Complete" dimensions not specified in `dims` with ones.
# Example: if N = 5, dims = (2, 3) and vals = (42, 12), this returns
# (1, 42, 12, 1, 1).
function complete_dims(::Val{N}, dims::Dims{M}, vals::Dims{M}) where {N, M}
    @assert N >= M
    i = 0
    vals_all = ntuple(Val(N)) do n
        if findfirst(dims .== n) === nothing
            1  # this dimension is not included in `dims`, so we put a 1
        else
            i += 1
            vals[i]
        end
    end
    @assert i == M
    vals_all :: Dims{N}
end

# Get axes (array regions) owned by all processes in a given pencil
# configuration.
function get_axes_matrix(decomp_dims::Dims{M},
                         proc_dims::Dims{M},
                         size_global::Dims{N}) where {N, M}
    axes = Array{ArrayRegion{N}, M}(undef, proc_dims)

    # Number of processes in every direction, including those where
    # decomposition is not applied.
    procs = complete_dims(Val(N), decomp_dims, proc_dims)

    for I in CartesianIndices(proc_dims)
        coords = complete_dims(Val(N), decomp_dims, Tuple(I))
        axes[I] = local_data_range.(coords, procs, size_global)
    end

    axes
end
