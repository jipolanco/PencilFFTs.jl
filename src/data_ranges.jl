# Functions determining local data ranges in the different pencil
# configurations.

# Get Cartesian coordinates of current process in communicator.
function _cart_coords(comm_cart) :: NTuple{3,Int}  # (1, p1, p2)
    maxdims = 3
    coords = MPI.Cart_coords(comm_cart, maxdims) .+ 1  # >= 1
    coords[1], coords[2], coords[3]
end

function _local_range(p, P, N)
    @assert 1 <= p <= P
    a = (N * (p - 1)) ÷ P + 1
    b = (N * p) ÷ P
    a:b
end

# Get local data range (Rx, Ry, Rz) in x-pencil configuration.
function _get_data_range_x(comm_cart, Nxyz, (P1, P2))
    Pxyz = 1, P1, P2
    ijk = _cart_coords(comm_cart)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz)
end

# Get local data range (Rx, Ry, Rz) in y-pencil configuration.
# Note that indices are permuted in this configuration!
function _get_data_range_y(comm_cart, Nxyz, (P1, P2))
    Pxyz = P1, 1, P2
    j, i, k = _cart_coords(comm_cart)
    ijk = (i, j, k)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz)
end

# Get local data range (Rx, Ry, Rz) in z-pencil configuration.
# Note that indices are permuted in this configuration!
function _get_data_range_z(comm_cart, Nxyz, (P1, P2))
    Pxyz = P1, P2, 1
    k, i, j = _cart_coords(comm_cart)
    ijk = (i, j, k)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz)
end
