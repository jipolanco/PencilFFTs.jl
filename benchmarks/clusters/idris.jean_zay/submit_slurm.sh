#!/bin/bash

# resolution=512
resolution=1024

repetitions=100

with_openmpi=0
intel_version=19.0.4

mkdir -p results details

if (( resolution == 512 )); then
    procs="128 256 512 1024 2048 4096 8192"
elif (( resolution == 1024 )); then
    procs="256 512 1024 2048 4096 8192 16384"
fi

module purge
module load intel-compilers/$intel_version
module load fftw/3.3.8
module load git
module load cmake/3.14.4
module load autoconf automake

if (( with_openmpi )); then
    mpi_label="openmpi"
    module load openmpi/4.0.1-cuda
else
    mpi_label="intel_${intel_version}"
    module load intel-mpi/$intel_version
fi

outfile_jl="results/PencilFFTs_N${resolution}_${mpi_label}.dat"
outfile_p3d="results/P3DFFT2_N${resolution}_${mpi_label}.dat"

# Compile p3dfft
export CC=icc
export CXX=icpc
export FC=ifort
srcdir="$(realpath -e ../../p3dfft)"
builddir=build.p3dfft.$mpi_label

if [[ ! -f $builddir/bench_p3dfft ]]; then
    mkdir -p $builddir
    pushd $builddir || exit 1
    cmake "$srcdir" -DCMAKE_BUILD_TYPE=Release || exit 2
    make -j4
    popd || exit 1
fi

for n in $procs; do
    echo "Submitting N = ${resolution} benchmark with $n processes..."
    sbatch <<EOF
#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=$n
# #SBATCH --ntasks-per-node=40
#SBATCH --time=30
#SBATCH --hint=nomultithread
#SBATCH --output="details/Nproc${n}_N${resolution}_${mpi_label}_%j.out"
#SBATCH --error="details/Nproc${n}_N${resolution}_${mpi_label}_%j.err"

#SBATCH --job-name=bench_N${resolution}
#SBATCH --dependency=singleton

module list

# Force precompilation of Julia packages in serial mode, to avoid race conditions.
julia -O3 -Cnative --project -e \
    'using Pkg; pkg"instantiate"; pkg"build"; pkg"precompile";
     using MPI; println("MPI: ", MPI.identify_implementation());'

# 1. Run PencilFFTs benchmark
srun julia -O3 -Cnative --check-bounds=no --compiled-modules=no --project \
    ../../benchmarks.jl -N $resolution -r 100 -o $outfile_jl || exit 2

# 2. Run P3DFFT benchmark
srun ./$builddir/bench_p3dfft $resolution $repetitions $outfile_p3d || exit 3

EOF
done
