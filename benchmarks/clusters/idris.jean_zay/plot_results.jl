#!/usr/bin/env julia

using DelimitedFiles: readdlm

import PyPlot
using LaTeXStrings
const plt = PyPlot
const mpl = PyPlot.matplotlib

struct Benchmark{HasParams, Style}
    name :: String
    filename_base :: String
    pyplot_style :: Style
    Benchmark(name, fname, has_params::Bool, style) =
        new{has_params, typeof(style)}(name, fname, style)
end

const MPI_TAG = "intel_19.0.4"

const RESOLUTIONS = (512, 1024)

const BENCH_PENCILS = Benchmark(
    "PencilFFTs", "results/PencilFFTs", true, (color=:C0, zorder=5),
)
const BENCH_P3DFFT = Benchmark(
    "P3DFFT", "results/P3DFFT2", false, (color=:C1, zorder=3),
)
const BENCH_LIST = (BENCH_PENCILS, BENCH_P3DFFT)

const STYLE_RESOLUTION = Dict(512 => (marker=:o, ),
                              1024 => (marker=:^, ))

const STYLE_IDEAL = (color=:black, ls=:dotted, label="ideal")

struct TransposeParams
    permute :: Bool
    isend :: Bool
    TransposeParams(; permute, isend) = new(permute, isend)
end

time_column(p::TransposeParams) = 2 * !p.permute + !p.isend + 1

time_columns(bench::Benchmark{true}, params) = time_column.(params)
time_columns(bench::Benchmark{false}, params) = (1, )

function plot_from_file!(ax, bench::Benchmark, resolution, params;
                         plot_ideal=false)
    filename = string(bench.filename_base, "_N$(resolution)_$(MPI_TAG).dat")
    data = readdlm(filename, Float64, comments=true) :: Matrix{Float64}
    Nxyz = data[:, 1:3]
    @assert all(Nxyz .== resolution)
    procs = data[:, 4]
    # proc_dims = @view data[:, 5:6]
    # repetitions = @view data[:, 7]

    cols = collect(7 .+ time_columns(bench, params))

    times = data[:, cols]
    times_ideal = similar(times)

    for j in axes(times, 2)
        α = times[1, j] * procs[1]
        for i in axes(times, 1)
            times_ideal[i, j] = α / procs[i]
        end
    end

    let st = STYLE_RESOLUTION[resolution]
        ax.plot(procs, times; st..., bench.pyplot_style...)
    end

    if plot_ideal
        ax.plot(procs, times_ideal[:, 1]; STYLE_IDEAL...)
    end

    ax
end

function plot_lib_comparison!(ax, benchs, resolution)
    params = (
        TransposeParams(permute=true, isend=true),
        # TransposeParams(permute=true, isend=false),
    )
    ax.set_xscale(:log, basex=2)
    ax.set_yscale(:log, basey=10)
    map(benchs) do bench
        plot_from_file!(
            ax, bench, resolution, params,
            plot_ideal = bench === BENCH_PENCILS,
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

function plot_lib_comparison()
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

plot_lib_comparison()

