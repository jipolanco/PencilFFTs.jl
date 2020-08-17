#!/bin/bash

# resolution=512
# resolution=1024
resolution=2048

repetitions=100

with_openmpi=0
intelmpi_version=2019.7

mkdir -p results details

if (( resolution == 512 )); then
    # procs="128 256 512 1024 2048 4096 8192"
    procs="1024 2048 4096 8192"
elif (( resolution == 1024 )); then
    procs="256 512 1024 2048 4096 8192 16384"
    # procs="512 1024 2048 4096 8192 16384"
elif (( resolution == 2048 )); then
    procs="512 1024 2048 4096 8192 16384"
fi

module purge
module load intel-compilers/19.1.1
module load fftw/3.3.8
module load git
module load cmake/3.18.0
module load autoconf automake

if (( with_openmpi )); then
    mpi_label="openmpi"
    module load openmpi/4.1.0rc1
else
    mpi_label="intelmpi_${intelmpi_version}"
    module load intel-mpi/$intelmpi_version
fi

export JULIA_MPI_BINARY=system

# Comment this to disable loading custom system image.
julia_sys=../../sys_benchmarks.so

if [[ -n $julia_sys ]]; then
    if [[ ! -f $julia_sys ]]; then
        echo "ERROR - system image not found: $julia_sys"
        exit 1
    fi
    julia_opt=(-O3 -Cnative -J$julia_sys --project)
else
    julia_opt=(-O3 -Cnative --project)
fi

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

outfile_jl="results/PencilFFTs_N${resolution}_${mpi_label}.dat"
outfile_p3d="results/P3DFFT2_N${resolution}_${mpi_label}.dat"

for n in $procs; do
    echo "Submitting N = ${resolution} benchmark with $n processes..."
    sbatch <<EOF
#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=$n
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --hint=nomultithread
#SBATCH --output="details/N${resolution}_Nproc${n}_${mpi_label}.out"
#SBATCH --error="details/N${resolution}_Nproc${n}_${mpi_label}.err"

#SBATCH --job-name=bench_N${resolution}
#SBATCH --dependency=singleton

module list

# Print version information
julia ${julia_opt[@]} -e \
    'using Pkg; using InteractiveUtils;
     pkg"instantiate"; pkg"precompile"; pkg"status"; versioninfo();
     using MPI; println("MPI: ", MPI.identify_implementation());'

# 1. Run PencilFFTs benchmark
srun julia ${julia_opt[@]} --check-bounds=no \
    ../../benchmarks.jl -N $resolution -r $repetitions -o $outfile_jl || exit 2

# 2. Run P3DFFT benchmark
srun ./$builddir/bench_p3dfft $resolution $repetitions $outfile_p3d || exit 3

EOF
done
