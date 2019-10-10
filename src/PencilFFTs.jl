module PencilFFTs

export PencilPlan

using MPI

const ArrayRegion = NTuple{3,UnitRange{Int}}

"""
    PencilPlan(comm::MPI.Comm, P1, P2, Nx, Ny, Nz)

Create "plan" for pencil-decomposed FFTs.

Data is decomposed among MPI processes in communicator `comm` (usually
`MPI_COMM_WORLD`), with `P1` and `P2` the number of processes in each of the
decomposed directions.

The real-space dimensions of the data to be transformed are `Nx`, `Ny` and `Nz`.
"""
struct PencilPlan
    # MPI communicator with Cartesian topology (describing the x-pencil layout).
    comm_cart_x :: MPI.Comm

    # Number of processes in the two decomposed directions.
    #   x-pencil: (P1, P2) = (Py, Pz)
    #   y-pencil: (P1, P2) = (Px, Pz)
    #   z-pencil: (P1, P2) = (Px, Py)
    P1 :: Int
    P2 :: Int

    # Global dimensions of real data (Nx, Ny, Nz).
    size_global :: Dims{3}

    # Local range of complex data, for each pencil configuration.
    crange_x :: ArrayRegion
    crange_y :: ArrayRegion
    crange_z :: ArrayRegion

    function PencilPlan(comm::MPI.Comm, P1, P2, Nx, Ny, Nz)
        Nproc = MPI.Comm_size(comm)

        if P1 * P2 != Nproc
            error("Decomposition with (P1, P2) = ($P1, $P2) not compatible with communicator size $Nproc.")
        end

        size_global = (Nx, Ny, Nz)

        if any(isodd.(size_global))
            # TODO Maybe this can be relaxed?
            error("Dimensions (Nx, Ny, Nz) must be even.")
        end

        # Create Cartesian communicators.
        comm_cart = let dims = [1, P1, P2]
            periods = [1, 1, 1]  # periodicity info is useful for MPI.Cart_shift
            reorder = true
            MPI.Cart_create(comm, dims, periods, reorder)
        end

        Nxyz_c = (_complex_size_x(Nx), Ny, Nz)  # global dimensions of complex data

        crange_x = _get_data_range_x(comm_cart, Nxyz_c, (P1, P2))
        crange_y = _get_data_range_y(comm_cart, Nxyz_c, (P1, P2))
        crange_z = _get_data_range_z(comm_cart, Nxyz_c, (P1, P2))

        new(comm_cart, P1, P2, size_global, crange_x, crange_y, crange_z)
    end
end

"Get Cartesian coordinates of current process in communicator."
function _cart_coords(comm_cart) :: NTuple{3,Int}  # (1, p1, p2)
    maxdims = 3
    coords = MPI.Cart_coords(comm_cart, maxdims) .+ 1  # >= 1
    coords[1], coords[2], coords[3]
end

"Length of first dimension when data is in complex space."
_complex_size_x(Nx) = (Nx >>> 1) + 1  # Nx/2 + 1

function _local_range(p, P, N)
    @assert 1 <= p <= P
    a = (N * (p - 1)) ÷ P + 1
    b = (N * p) ÷ P
    a:b
end

"Get local range of complex data in x-pencil configuration."
function _get_data_range_x(comm_cart, Nxyz_c, (P1, P2))
    Pxyz = 1, P1, P2
    ijk = _cart_coords(comm_cart)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz_c)
end

function _get_data_range_y(comm_cart, Nxyz_c, (P1, P2))
    Pxyz = P1, 1, P2
    j, i, k = _cart_coords(comm_cart)
    ijk = (i, j, k)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz_c)
end

function _get_data_range_z(comm_cart, Nxyz_c, (P1, P2))
    Pxyz = P1, P2, 1
    k, i, j = _cart_coords(comm_cart)
    ijk = (i, j, k)
    @assert all(1 .<= ijk .<= Pxyz)
    _local_range.(ijk, Pxyz, Nxyz_c)
end

end # module
