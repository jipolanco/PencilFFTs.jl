#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from typing import NamedTuple

MPI_TAG = "intel_19.0.4"


class Benchmark(NamedTuple):
    name: str
    filename_fmt: str
    plot_style: dict = dict()


BENCH_PENCILS = Benchmark('PencilFFTs', 'results/PencilFFTs_N{N}_{MPI}.dat',
                          plot_style=dict(color='C0'))

BENCH_P3DFFT = Benchmark('P3DFFT', 'results/P3DFFT3_N{N}_{MPI}.dat',
                         plot_style=dict(color='C1'))


RESOLUTIONS = (512, 1024)

PARAMS_PI = 0  # with dimension permutations + Isend/Irecv
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

STYLE_PERM = dict(color='C0')
STYLE_NOPERM = dict(color='C3')
STYLE_ISEND = dict(ls='-')
STYLE_ALLTO = dict(ls='--')

STYLE_PARAMS = (
    dict(**STYLE_PERM, **STYLE_ISEND),
    dict(**STYLE_PERM, **STYLE_ALLTO),
    dict(**STYLE_NOPERM, **STYLE_ISEND),
    dict(**STYLE_NOPERM, **STYLE_ALLTO),
)

STYLE_RESOLUTION = {
    512: dict(marker='o'),
    1024: dict(marker='^')
}

STYLE_IDEAL = dict(color='black', ls=':', label='ideal')


def plot_from_file(ax: plt.Axes, bench: Benchmark, resolution: int,
                   speedup=False, params=PARAMS_ALL, plot_kw=None):
    ax.set_xscale('log', basex=2)
    ax.set_yscale('log', basey=(2 if speedup else 10))

    is_pencilffts = bench is BENCH_PENCILS
    filename = bench.filename_fmt.format(N=resolution, MPI=MPI_TAG)
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

    for n in range(Ntimes):
        kw = bench.plot_style.copy()
        if is_pencilffts:
            if n not in params:
                continue
            if len(params) > 1:
                kw.update(STYLE_PARAMS[n])
        if plot_kw is not None:
            kw.update(plot_kw)
        ax.plot(procs, times[:, n], **kw)

    if is_pencilffts:
        ax.plot(procs, times_ideal[:, 0], **STYLE_IDEAL)


def compare_params(ax: plt.Axes, resolution, bench=BENCH_PENCILS,
                   params=PARAMS_ALL, plot_kw=None):
    filename = bench.filename_fmt.format(N=resolution, MPI=MPI_TAG)
    data = np.loadtxt(filename)
    Nxyz = data[:, 0:3]
    N = Nxyz[0, 0]
    assert np.all(Nxyz == N)  # they're all the same
    procs = data[:, 3]
    # proc_dims = data[:, 4:6]
    times = data[:, 7:]
    Ntimes = times.shape[1]

    # Compare with the first column (default params)
    times /= times[:, 0:1]

    for n in params:
        kw = STYLE_PARAMS[n].copy()
        if plot_kw is not None:
            kw.update(plot_kw)
        ax.plot(procs, times[:, n], **kw)


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


def legend_params(ax: plt.Axes, params):
    styles = map(lambda p: STYLE_PARAMS[p], params)


def legend_resolution(ax: plt.Axes, resolutions, loc='lower left'):
    styles = [dict(color='black', ls='-', **STYLE_RESOLUTION[n])
              for n in resolutions]
    labels = map(lambda N: f'{N}³', resolutions)
    add_legend(ax, styles, labels, loc=loc, title='Resolution')


def plot_comparison(ax: plt.Axes, N, legend=False, **kwargs):
    st = STYLE_RESOLUTION[N]
    bench = (BENCH_PENCILS, BENCH_P3DFFT)
    for b in bench:
        plot_from_file(ax, b, N, params=(PARAMS_DEFAULT, ),
                       plot_kw=st, **kwargs)
    if legend:
        legend_benchmarks(ax, bench)


def plot_params(ax: plt.Axes, N, legend=False, **kwargs):
    ax.set_xscale('log', basex=2)
    st = STYLE_RESOLUTION[N]
    params = (PARAMS_PI, PARAMS_PA)
    compare_params(ax, N, BENCH_PENCILS, params, plot_kw=st)


def plot_timings(ax: plt.Axes, resolutions, plot_func=plot_comparison,
                 speedup=False):
    for n, N in enumerate(resolutions):
        plot_func(ax, N, speedup=speedup, legend=(n == 0))

    if speedup:
        ylab = 'Speedup'
        loc = 'upper left'
    else:
        ylab = 'Time (milliseconds)'
        loc = 'lower left'

    legend_resolution(ax, resolutions, loc)
    ax.set_xlabel('MPI processes')

    # Show '1024' instead of '2^10'
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(mpl.ticker.ScalarFormatter())

    ax.set_ylabel(ylab)


fig, ax = plt.subplots()
plot_timings(ax, RESOLUTIONS, plot_comparison)
fig.savefig('timing_comparison.svg')

fig, ax = plt.subplots()
plot_timings(ax, RESOLUTIONS, plot_params)
# fig.savefig('timing_params.svg')

plt.show()
