#!/usr/bin/env bash

using PackageCompiler

precompile_script =
"""
using PencilFFTs
using PencilFFTs.PencilArrays

import FFTW
using MPI

using ArgParse

MPI.Init()
comm = MPI.COMM_WORLD

s = ArgParseSettings()
@add_arg_table! s begin end
parse_args(s)

plan = PencilFFTPlan((32, 32, 32), Transforms.RFFT(), (1, 1), comm)
u = allocate_input(plan)
v = plan * u
w = plan \\ v
"""

mktemp() do fname, io
    write(io, precompile_script)
    close(io)
    @time create_sysimage(
        [:MPI, :FFTW, :PencilFFTs, :ArgParse],
        sysimage_path="sys_benchmarks.so",
        cpu_target="native",
        precompile_execution_file=fname,
    )
end

