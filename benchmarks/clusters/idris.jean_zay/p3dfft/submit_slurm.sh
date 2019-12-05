#!/bin/bash

resolution=1024
outfile="p3dfft_N${resolution}.dat"
repetitions=100

module purge
module load intel-all/19.0.4
module load git
module load cmake/3.14.4

export CXX=icpc

srcdir="$(realpath -e ../../../p3dfft.3)"
builddir=build

mkdir -p $builddir

pushd $builddir || exit 1
cmake "$srcdir" -DCMAKE_BUILD_TYPE=Release \
    -DP3DFFT_CXX_FLAGS="-O3 -DNDEBUG -xHost" \
    -DUSE_JULIA_FFTW=ON || exit 2
make -j4
popd || exit 1


for n in 256 512 1024 2048 4096 8192 16384; do
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

#SBATCH --job-name=bench_p3dfft_N${resolution}
#SBATCH --dependency=singleton

srun ./build/bench_p3dfft $resolution $repetitions $outfile
EOF
done
