#!/bin/bash

# resolution=512
# resolution=1024
resolution=2048

repetitions=100

with_openmpi=0
intelmpi_version=2019.9
openmpi_version=4.0.5

mkdir -p results

if (( resolution == 256 )); then
    procs="64 128 256 512 1024 2048 4096"
elif (( resolution == 512 )); then
    procs="128 256 512 1024 2048 4096 8192"
elif (( resolution == 1024 )); then
    procs="256 512 1024 2048 4096 8192"
elif (( resolution == 2048 )); then
    procs="512 1024 2048 4096 8192 16384"
fi

module purge

if (( with_openmpi )); then
    mpi_label="OpenMPI.${openmpi_version}"
    module load openmpi/$openmpi_version
    module load intel-compilers/19.1.1
else
    mpi_label="IntelMPI.${intelmpi_version}"
    module load intel-mpi/$intelmpi_version
    module load intel-compilers/19.1.3
fi

# module load julia/1.5.2  # use cluster installation
module load fftw/3.3.8
module load git
module load cmake
module load autoconf automake

export JULIA_MPI_BINARY=system

# Comment this to disable loading custom system image.
# julia_sys=../../sys_benchmarks.so

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
    make -j4 || exit 3
    popd || exit 1
fi

outdir="results/${mpi_label}/N${resolution}"
mkdir -pv "$outdir"

outfile_jl="$outdir/PencilFFTs.dat"
outfile_p3d="$outdir/P3DFFT2.dat"

julia ${julia_opt[@]} -e 'using Pkg; Pkg.build("MPI", verbose=true)'

export PENCILFFTS_BENCH_REPETITIONS=$repetitions
export PENCILFFTS_BENCH_DIMENSIONS=$resolution
export PENCILFFTS_BENCH_OUTPUT=$outfile_jl

for n in $procs; do
    echo "Submitting N = ${resolution} benchmark with $n processes..."
    sbatch <<EOF
#!/bin/bash

#SBATCH --exclusive
#SBATCH --contiguous
#SBATCH --ntasks=$n
#SBATCH --ntasks-per-node=40
#SBATCH --time=1:00:00
#SBATCH --hint=nomultithread
#SBATCH --output="${outdir}/proc_${n}.out"
#SBATCH --error="${outdir}/proc_${n}.err"

#SBATCH --job-name=bench_N${resolution}
#SBATCH --dependency=singleton

module list

# Print version information
julia ${julia_opt[@]} -e \
    'using Pkg; using InteractiveUtils;
     pkg"instantiate"; pkg"precompile"; pkg"status"; versioninfo();
     using MPI; println("MPI: ", MPI.identify_implementation());'

# 1. Run PencilFFTs benchmark
srun julia ${julia_opt[@]} --check-bounds=no ../../benchmarks.jl  || exit 2

# 2. Run P3DFFT benchmark
srun ./$builddir/bench_p3dfft $resolution $repetitions $outfile_p3d || exit 3

EOF
done
