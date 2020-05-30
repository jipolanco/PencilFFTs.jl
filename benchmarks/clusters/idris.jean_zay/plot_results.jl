#!/usr/bin/env julia

using DelimitedFiles: readdlm

import PyPlot
using LaTeXStrings
const plt = PyPlot
const mpl = PyPlot.matplotlib

struct Benchmark{Style}
    name   :: String
    filename_base :: String
    filename_suffix :: String
    pyplot_style :: Style
    Benchmark(name, fname, style; suffix="") =
        new{typeof(style)}(name, fname, suffix, style)
end

const MPI_TAG = "intel_19.0.4"

const RESOLUTIONS = (512, 1024)

const BENCH_PENCILS = Benchmark(
    "PencilFFTs", "results/PencilFFTs", (color=:C0, zorder=5),
    suffix="_PI",
)
const BENCH_P3DFFT = Benchmark(
    "P3DFFT", "results/P3DFFT2", (color=:C1, zorder=3),
)
const BENCH_LIST = (BENCH_PENCILS, BENCH_P3DFFT)

const STYLE_RESOLUTION = Dict(512 => (marker=:o, ),
                              1024 => (marker=:^, ))

const STYLE_IDEAL = (color=:black, ls=:dotted, label="ideal")

function load_timings(bench::Benchmark, resolution)
    filename = string(bench.filename_base,
                      "_N$(resolution)_$(MPI_TAG)$(bench.filename_suffix).dat")
    data = readdlm(filename, Float64, comments=true) :: Matrix{Float64}
    Nxyz = data[:, 1:3]
    @assert all(Nxyz .== resolution)
    procs = data[:, 4]
    proc_dims = data[:, 5:6]
    repetitions = data[:, 7]
    times = data[:, 8]
    (
        Nxyz = Nxyz,
        procs = procs,
        proc_dims = proc_dims,
        repetitions = repetitions,
        times = times,
    )
end

function plot_from_file!(ax, bench::Benchmark, resolution; plot_ideal=false)
    data = load_timings(bench, resolution)

    st = STYLE_RESOLUTION[resolution]
    ax.plot(data.procs, data.times; st..., bench.pyplot_style...)

    if plot_ideal
        t = data.times
        p = data.procs
        t_ideal = similar(t)
        for n in eachindex(t)
            t_ideal[n] = t[1] * p[1] / p[n]
        end
        p, t_ideal
        ax.plot(p, t_ideal; STYLE_IDEAL...)
    end

    ax
end

function plot_lib_comparison!(ax, benchs, resolution)
    ax.set_xscale(:log, basex=2)
    ax.set_yscale(:log, basey=10)
    map(benchs) do bench
        plot_from_file!(
            ax, bench, resolution, plot_ideal = bench === BENCH_PENCILS,
        )
    end
    ax
end

function legend_libs!(ax, benchs; with_ideal=false)
    styles = Any[getfield.(benchs, :pyplot_style)...]
    labels = Any[getfield.(benchs, :name)...]
    if with_ideal
        push!(styles, STYLE_IDEAL)
        push!(labels, "Ideal")
    end
    draw_legend!(ax, styles, labels, loc="upper right")
end

function legend_resolution!(ax, resolutions)
    styles = [(color=:black, ls=:solid, STYLE_RESOLUTION[n]...)
              for n in resolutions]
    labels = map(n -> "$(n)³", resolutions)
    draw_legend!(ax, styles, labels, loc="lower left", title="Resolution")
end

function draw_legend!(ax, styles, labels; kwargs...)
    leg = ax.get_legend()
    lines = [mpl.lines.Line2D(Float64[], Float64[]; style...)
             for style in styles]
    ax.legend(lines, labels; kwargs...)
    if leg !== nothing
        ax.add_artist(leg)
    end
    ax
end

function plot_timings()
    resolutions = RESOLUTIONS
    benchs = BENCH_LIST
    fig, ax = plt.subplots()

    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Time (milliseconds)")

    map(resolutions) do N
        plot_lib_comparison!(ax, benchs, N)
    end

    legend_libs!(ax, benchs, with_ideal=true)
    legend_resolution!(ax, resolutions)

    # Show '1024' instead of '2^10'
    for axis in (ax.xaxis, ax.yaxis)
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())
    end

    ax.set_ylim(top=1.05e3)

    fig.savefig("timing_comparison.svg")

    fig
end

function plot_relative_times()
    resolutions = RESOLUTIONS
    benchs = BENCH_LIST
    fig, ax = plt.subplots()

    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Relative time")
end

plot_timings()
