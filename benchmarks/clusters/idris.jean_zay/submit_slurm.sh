#!/bin/bash

resolution=1024
outfile="jean_zay_N${resolution}.dat"

module purge
module load intel-all/19.0.4

for n in 256 512 1024 2048 4096 8192 16384; do
    echo "Launching benchmark with $n processes..."
    srun --exclusive --ntasks=$n -t 30 \
        --ntasks-per-node=40 \
        --hint=nomultithread \
        --output="Nproc${n}_N${resolution}.out.%j.log" \
        --error="Nproc${n}_N${resolution}.err.%j.log" \
        julia --project=. ./benchmarks.jl \
        -N $resolution \
        -r 100 \
        -o $outfile || exit 2
done
