#!/usr/bin/env julia

using DelimitedFiles: readdlm

import PyPlot
using LaTeXStrings
const plt = PyPlot
const mpl = PyPlot.matplotlib

struct Benchmark{Style}
    name     :: String
    filename :: String
    pyplot_style :: Style
    Benchmark(name, fname, style) = new{typeof(style)}(name, fname, style)
end

struct TimerData
    avg :: Vector{Float64}
    std :: Vector{Float64}
    min :: Vector{Float64}
    max :: Vector{Float64}
end

const MPI_TAG = Ref("IntelMPI.2019.9")

const STYLE_IDEAL = (color=:black, ls=:dotted, label="ideal")

function load_timings(bench::Benchmark, resolution)
    filename = joinpath("results", MPI_TAG[], "N$resolution", bench.filename)
    data = readdlm(filename, Float64, comments=true) :: Matrix{Float64}
    Nxyz = data[:, 1:3]
    @assert all(Nxyz .== resolution)
    procs = data[:, 4]
    proc_dims = data[:, 5:6]
    repetitions = data[:, 7]
    times = TimerData((data[:, j] for j = 8:11)...)
    (
        Nxyz = Nxyz,
        procs = procs,
        proc_dims = proc_dims,
        repetitions = repetitions,
        times = times,
    )
end

function plot_from_file!(ax, bench::Benchmark, resolution;
                         plot_ideal=false, error_bars=nothing)
    data = load_timings(bench, resolution)
    times = data.times
    t = times.avg

    ax.plot(data.procs, t; bench.pyplot_style...)

    colour = bench.pyplot_style.color
    if error_bars == :extrema
        ax.errorbar(data.procs, t; color=colour)
    elseif error_bars == :std
        δ = times.std ./ 2
        ax.fill_between(data.procs, t .- δ, t .+ δ; alpha=0.2, color=colour)
    end

    if plot_ideal
        plot_ideal_scaling!(ax, data, t)
        add_text_resolution!(ax, resolution, data.procs, t)
    end

    ax
end

function plot_ideal_scaling!(ax, data, t)
    p = data.procs
    t_ideal = similar(t)
    for n in eachindex(t)
        t_ideal[n] = t[1] * p[1] / p[n]
    end
    ax.plot(p, t_ideal; STYLE_IDEAL...)
end

function add_text_resolution!(ax, N, xs, ys)
    x = first(xs)
    y = first(ys)
    if N == 512
        kws = (ha = :left, va = :top)
        x *= 0.95
        y *= 0.65
    else
        kws = (ha = :right, va = :center)
        x *= 0.9
        y *= 1.02
    end
    ax.text(
        x, y, latexstring("$N^3");
        fontsize = "large",
        kws...
    )
end

function plot_lib_comparison!(ax, benchs, resolution)
    ax.set_xscale(:log, base=2)
    ax.set_yscale(:log, base=10)
    map(benchs) do bench
        plot_ideal = bench === first(benchs)
        plot_from_file!(
            ax, bench, resolution;
            plot_ideal,
            # error_bars = :std,
        )
    end
    ax
end

function legend_libs!(ax, benchs; with_ideal=false, outside=false)
    styles = Any[getfield.(benchs, :pyplot_style)...]
    labels = Any[getfield.(benchs, :name)...]
    if with_ideal
        push!(styles, STYLE_IDEAL)
        push!(labels, "Ideal")
    end
    kws = if outside
        (loc = "center left", bbox_to_anchor = (1.0, 0.5))
    else
        (loc = "lower left", )
    end
    draw_legend!(ax, styles, labels; frameon=false, kws...)
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

# Wrap matplotlib's SVG writer.
struct SVGWriter{File<:IO} <: IO
    fh :: File
end

Base.isreadable(io::SVGWriter) = isreadable(io.fh)
Base.iswritable(io::SVGWriter) = iswritable(io.fh)
Base.isopen(io::SVGWriter) = isopen(io.fh)

function Base.write(io::SVGWriter, s::Union{SubString{String}, String})
    # We remove the image height and width from the header.
    # This way the SVG image takes all available space in browsers.
    p = "\"\\S+pt\""  # "316.8pt"
    pat = Regex("(<svg .*)height=$p (.*) width=$p")
    rep = replace(s, pat => s"\1\2")
    write(io.fh, rep)
end

function plot_timings()
    resolutions = (
        # 256,
        512,
        1024,
        2048,
    )
    style = (fillstyle = :none, ms = 7, markeredgewidth = 1.5, linewidth = 1.5)
    benchs = (
        Benchmark(
            "PencilFFTs (default)", "PencilFFTs_PI.dat",
            (style..., color = :C0, marker = :o, zorder = 5),
        ),
        Benchmark(
            "PencilFFTs (Alltoallv)", "PencilFFTs_PA.dat",
            (style..., color = :C2, marker = :s, zorder = 8,
             linestyle = :dashed, linewidth = 1.0),
        ),
        Benchmark(
            "P3DFFT", "P3DFFT2.dat",
            (style..., color = :C1, marker = :x, zorder = 4, markeredgewidth = 2),
        ),
    )

    fig = plt.figure(figsize = (6, 4.2) .* 1.1, dpi = 150)
    ax = fig.subplots()

    ax.set_xlabel("MPI processes")
    ax.set_ylabel("Time (milliseconds)")

    map(resolutions) do N
        plot_lib_comparison!(ax, benchs, N)
    end

    legend_libs!(ax, benchs, with_ideal=true, outside=false)

    # Show '1024' instead of '2^10'
    for axis in (ax.xaxis, ax.yaxis)
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())
    end

    open("timing_comparison.svg", "w") do ff
        io = SVGWriter(ff)
        @time fig.savefig(io)
    end

    fig
end

plot_timings()
