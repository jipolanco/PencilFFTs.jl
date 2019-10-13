# Functions determining local data ranges in the different pencil
# configurations.

function local_data_range(p, P, N)
    @assert 1 <= p <= P
    a = (N * (p - 1)) ÷ P + 1
    b = (N * p) ÷ P
    a:b
end

# Get axes (array regions) owned by all processes in a given pencil
# configuration.
function get_axes_matrix(::Val{1}, dims::Dims{2}, Nxyz::Dims{3})
    axes = Matrix{ArrayRegion{3}}(undef, dims...)
    Pxyz = (1, dims...)
    for I in CartesianIndices(dims)
        coords = (1, Tuple(I)...)  # 3D coordinates
        axes[I] = local_data_range.(coords, Pxyz, Nxyz)
    end
    axes
end

function get_axes_matrix(::Val{2}, dims::Dims{2}, Nxyz::Dims{3})
    axes = Matrix{ArrayRegion{3}}(undef, dims...)
    Pxyz = (dims[1], 1, dims[2])
    for I in CartesianIndices(dims)
        cx, cz = Tuple(I)
        coords = (cx, 1, cz)  # 3D coordinates
        axes[I] = local_data_range.(coords, Pxyz, Nxyz)
    end
    axes
end

function get_axes_matrix(::Val{3}, dims::Dims{2}, Nxyz::Dims{3})
    axes = Matrix{ArrayRegion{3}}(undef, dims...)
    Pxyz = (dims..., 1)
    for I in CartesianIndices(dims)
        coords = (Tuple(I)..., 1)  # 3D coordinates
        axes[I] = local_data_range.(coords, Pxyz, Nxyz)
    end
    axes
end
