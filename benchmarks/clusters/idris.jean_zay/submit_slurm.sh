#!/bin/bash

# resolution=512
resolution=1024
repetitions=100

outfile_jl="PencilFFTs_N${resolution}.dat"
outfile_p3d="P3DFFT3_N${resolution}.dat"

if (( resolution == 512 )); then
    procs="128 256 512 1024 2048 4096"
elif (( resolution == 1024 )); then
    procs="256 512 1024 2048 4096 8192 16384"
fi

module purge
module load intel-all/19.0.4
module load git
module load cmake/3.14.4

# Compile p3dfft
export CXX=icpc
srcdir="$(realpath -e ../../p3dfft.3)"
builddir=build.p3dfft

if [[ ! -d $builddir ]]; then
    mkdir -p $builddir
    pushd $builddir || exit 1
    cmake "$srcdir" -DCMAKE_BUILD_TYPE=Release \
        -DP3DFFT_CXX_FLAGS="-O3 -DNDEBUG -xHost" \
        -DUSE_JULIA_FFTW=ON || exit 2
    make -j4
    popd || exit 1
fi

for n in $procs; do
    echo "Submitting N = ${resolution} benchmark with $n processes..."
    sbatch <<EOF
#!/bin/bash

#SBATCH --exclusive
#SBATCH --ntasks=$n
#SBATCH --ntasks-per-node=40
#SBATCH --time=30
#SBATCH --hint=nomultithread
#SBATCH --output="Nproc${n}_N${resolution}.out.%j.log"
#SBATCH --error="Nproc${n}_N${resolution}.err.%j.log"

#SBATCH --job-name=bench_N${resolution}
#SBATCH --dependency=singleton

# https://discourse.julialang.org/t/precompilation-error-using-hpc/17094
# https://discourse.julialang.org/t/run-a-julia-application-at-large-scale-on-thousands-of-nodes/23873

# Force precompilation of Julia modules in serial mode, to workaround issues
# described in the links above.
echo "Precompiling Julia modules..."
outfile_pre=\$(mktemp -u)
julia -O3 --check-bounds=no --project \
    ../../benchmarks.jl -N 8 -r 2 -o "\$outfile_pre" > /dev/null || exit 2

# 1. Run PencilFFTs benchmark
srun julia -O3 --check-bounds=no --project \
    ../../benchmarks.jl -N $resolution -r 100 -o $outfile_jl || exit 2

# 2. Run P3DFFT benchmark
srun ./$builddir/bench_p3dfft $resolution $repetitions $outfile_p3d || exit 3

EOF
done
