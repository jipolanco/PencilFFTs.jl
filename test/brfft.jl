# Test real-to-complex FFT using a backwards plan (BRFFT).

using PencilFFTs
using MPI
using LinearAlgebra
using FFTW
using AMDGPU
using CUDA
using GPUArrays

using Test

AT = [Array,]
if CUDA.functional()
    push!(AT, CuArray)
end
if AMDGPU.functional(:rocfft)
    push!(AT, ROCArray)
end

MPI.Init()
comm = MPI.COMM_WORLD

let dev_null = @static Sys.iswindows() ? "nul" : "/dev/null"
    MPI.Comm_rank(comm) == 0 || redirect_stdout(open(dev_null, "w"))
end

for type ∈ AT
    @testset "BRFFT: odd = $odd, type = $type" for odd ∈ (false, true)
        dims_real = (12, 13, 16 + odd)  # dimensions in physical (real) space
        dims_coef = (12, 13, 9)         # dimensions in coefficient (complex) space
        pen = Pencil(type, dims_coef, comm)

        uc = PencilArray{ComplexF64}(undef, pen)
        uc .= 0
        @allowscalar begin
            uc[2, 4, 3] = 1 + 2im
        end

        plan_c2r = PencilFFTPlan(uc, Transforms.BRFFT(dims_real))
        @test size(plan_c2r) == dims_coef  # = size of input

        ur = plan_c2r * uc
        @test size_global(ur, LogicalOrder()) == dims_real

        # Equivalent using FFTW.
        # Note that by default FFTW performs the c2r transform along the first
        # dimension, while PencilFFTs does it along the last one.
        uc_fftw = gather(uc)
        ur_full = gather(ur)
        if uc_fftw !== nothing
            bfft!(uc_fftw, (1, 2))
            ur_fftw = brfft(uc_fftw, dims_real[end], 3)
            @test ur_full ≈ ur_fftw
        end

        # Check normalisation
        uc_back = plan_c2r \ ur
        @test isapprox(uc_back, uc; atol = 1e-8)
    end
end