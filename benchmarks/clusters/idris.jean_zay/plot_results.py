#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import NamedTuple


class Benchmark(NamedTuple):
    name: str
    filename_fmt: str
    plot_style: dict = dict()


BENCH_PENCILS = Benchmark('PencilFFTs', 'results/jean_zay_N{N}.dat',
                          plot_style=dict(color='C0'))

BENCH_P3DFFT = Benchmark('P3DFFT', 'p3dfft/results/p3dfft_N{N}.dat',
                         plot_style=dict(color='C1'))


RESOLUTIONS = (512, 1024)

PARAMS_PI = 0  # with dimension permtuations + Isend/Irecv
PARAMS_PA = 1
PARAMS_NI = 2
PARAMS_NA = 3  # no dimension permutations + Alltoallv
PARAMS_DEFAULT = PARAMS_PI

PARAMS_ALL = (
    PARAMS_PI,
    PARAMS_PA,
    PARAMS_NI,
    PARAMS_NA,
)

STYLE_PARAMS = (
    dict(color='C0'),
    dict(color='C1'),
    dict(color='C2'),
    dict(color='C3'),
)

STYLE_RESOLUTION = {
    512: dict(marker='o'),
    1024: dict(marker='^')
}

STYLE_IDEAL = dict(color='black', ls=':', label='ideal')


def plot_from_file(ax: plt.Axes, bench: Benchmark, resolution: int,
                   speedup=False, params=PARAMS_ALL, plot_kw=None):
    is_pencilffts = bench is BENCH_PENCILS
    filename = bench.filename_fmt.format(N=resolution)
    data = np.loadtxt(filename)
    Nxyz = data[:, 0:3]
    N = Nxyz[0, 0]
    assert np.all(Nxyz == N)  # they're all the same
    procs = data[:, 3]
    # proc_dims = data[:, 4:6]
    times = data[:, 7:]
    Ntimes = times.shape[1]

    times_ideal = np.outer(1.0 / procs[:], times[0, :]) * procs[0]

    if speedup:
        times[:, :] = times[0, :] / times[:, :]
        times_ideal[:, :] = times_ideal[0, :] / times_ideal[:, :]

    lines = []

    for n in range(Ntimes):
        kw = bench.plot_style.copy()
        if is_pencilffts:
            if n not in params:
                continue
            if len(params) > 1:
                kw.update(STYLE_PARAMS[n])
        if plot_kw is not None:
            kw.update(plot_kw)
        lines += ax.plot(procs, times[:, n], **kw)

    if is_pencilffts:
        lines += ax.plot(procs, times_ideal[:, 0], **STYLE_IDEAL)

    return lines


def add_legend(ax: plt.Axes, styles, labels, **kwargs):
    leg = ax.get_legend()
    lines = [mpl.lines.Line2D([], [], **style) for style in styles]
    ax.legend(lines, labels, **kwargs)
    if leg is not None:
        ax.add_artist(leg)


def legend_benchmarks(ax: plt.Axes, bench):
    styles = [*map(lambda b: b.plot_style, bench), STYLE_IDEAL]
    labels = [*map(lambda b: b.name, bench), 'Ideal']
    add_legend(ax, styles, labels, loc='upper right')


def legend_resolution(ax: plt.Axes, resolutions):
    styles = [dict(color='black', ls='-', **STYLE_RESOLUTION[n])
              for n in resolutions]
    labels = resolutions
    add_legend(ax, styles, labels, loc='lower left',
               title='Resolution')


def plot_comparison(ax: plt.Axes, N, legend=False, **kwargs):
    st = STYLE_RESOLUTION[N]
    bench = (BENCH_PENCILS, BENCH_P3DFFT)
    for b in bench:
        plot_from_file(ax, b, N, params=(PARAMS_DEFAULT, ),
                       plot_kw=st, **kwargs)
    if legend:
        legend_benchmarks(ax, bench)


def plot_timings(ax: plt.Axes, resolutions, comparison=True, speedup=False):
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=(2 if speedup else 10))

    for n, N in enumerate(resolutions):
        if comparison:
            plot_comparison(ax, N, speedup=speedup, legend=(n == 0))

    legend_resolution(ax, resolutions)
    ax.set_xlabel('MPI processes')

    # Show '1024' instead of '2^10'
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ylab = 'Speedup' if speedup else 'Time (milliseconds)'
    ax.set_ylabel(ylab)


fig, ax = plt.subplots()
plot_timings(ax, RESOLUTIONS, comparison=True)
fig.savefig('timing_comparison.svg')

plt.show()
