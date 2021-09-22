#!/bin/python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np


def fast_kde(x, bw=4.5):
    """
    A fft-based Gaussian kernel density estimate (KDE)
    The code was adapted from https://github.com/mfouesneau/faststats

    Parameters
    ----------
    x : Numpy array or list
    bw : float
        Bandwidth scaling factor for the KDE. Should be larger than 0. The higher this number the
        smoother the KDE will be. Defaults to 4.5 which is essentially the same as the Scott's rule
        of thumb (the default rule used by SciPy).

    Returns
    -------
    density: A gridded 1D KDE of the input points (x)
    xmin: minimum value of x
    xmax: maximum value of x
    """

    from scipy.signal import gaussian, convolve
    from scipy.stats import entropy

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    nx = 200

    xmin, xmax = np.min(x), np.max(x)

    dx = (xmax - xmin) / (nx - 1)
    std_x = entropy((x - xmin) / dx) * bw
    if ~np.isfinite(std_x):
        std_x = 0.0
    grid, _ = np.histogram(x, bins=nx)

    scotts_factor = n ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * std_x)
    kernel = gaussian(kern_nx, scotts_factor * std_x)

    npad = min(nx, 2 * kern_nx)
    grid = np.concatenate([grid[npad:0:-1], grid, grid[nx : nx - npad : -1]])
    density = convolve(grid, kernel, mode="same")[npad : npad + nx]

    norm_factor = n * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    density = density / norm_factor

    return density, xmin, xmax


def kdeplot_op(ax, data, bw, prior=None, prior_alpha=1, prior_style="--"):
    """Get a list of density and likelihood plots, if a prior is provided."""
    ls = []
    pls = []
    errored = []
    density, l, u = fast_kde(data, bw)
    x = np.linspace(l, u, len(density))
    if prior is not None:

        try:
            p = prior.logpdf(x)
        except ValueError:
            p = x.copy()
            for i, xi in enumerate(x):
                p[i] = prior.logpdf(xi)
        pls.append(ax.plot(x, np.exp(p), alpha=prior_alpha, ls=prior_style))

    ls.append(ax.plot(x, density))

    if errored:
        ax.text(
            0.27,
            0.47,
            "WARNING: KDE plot failed for: " + ",".join(errored),
            bbox={"facecolor": "red", "alpha": 0.5, "pad": 10},
            style="italic",
        )

    return ls, pls


def get_axis(ax, default_rows, default_columns, **default_kwargs):
    """Verifies the provided axis is of the correct shape, and creates one if needed.

    Args:
        ax: matplotlib axis or None
        default_rows: int, expected rows in axis
        default_columns: int, expected columns in axis
        **default_kwargs: keyword arguments to pass to plt.subplot

    Returns:
        axis, or raises an error
    """

    default_shape = (default_rows, default_columns)
    if ax is None:
        fig, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError("Subplots with shape %r required" % (default_shape,))
    return fig, ax


def traceplot(
    trace,
    varnames,
    tune,
    figsize=None,
    combined=False,
    plots_per_fig=3,
    priors=None,
    draw_each_trace=True,
    bw=4.5,
    text_size=None,
    display_additinal_info=False,
    **kwargs
):

    # inspired by pymc3 with kisses

    if figsize is None:
        figsize = 9, plots_per_fig * 2.5

    width = trace.shape[0]
    tune = width - tune

    iterablator = lambda x: [x] if len(x) - 1 else x

    if priors is None:
        priors = [None] * len(varnames)

    custom_lines_hist = [
        Line2D([0], [0], linestyle="-", color="C0", lw=1),
        Line2D([0], [0], linestyle="--", color="C1", lw=1),
    ]

    custom_lines_trace = [
        Line2D([0], [0], linestyle="-", color="maroon", lw=1),
        Line2D([0], [0], linestyle="-", color="C0", lw=1),
    ]

    axs = []
    figs = []
    subfigs = []

    for chunk in range(0, len(varnames), plots_per_fig):

        eff_ppf = min(plots_per_fig,len(varnames[chunk:]))
        eff_figsize = figsize[0], figsize[1]*eff_ppf/plots_per_fig

        figs.append(plt.figure(constrained_layout=True, figsize=eff_figsize))
        subfig = figs[-1].subfigures(eff_ppf, 1, wspace=0.07)

        if eff_ppf == 1:
            subfig = [subfig]

        subfigs.append(subfig)

        for i in range(eff_ppf):

            axs.append(subfigs[-1][i].subplots(1, 2))
            data = trace[..., chunk + i]

            posterior = data[-tune:].flatten()
            plot_posterior_op(
                posterior,
                ax=axs[-1][0],
                bw=bw,
                prior=priors[chunk + i],
                text_size=scale_text(figsize, text_size),
                display_additinal_info=display_additinal_info,
                **kwargs
            )
            if draw_each_trace:
                axs[-1][1].plot(
                    range(0, tune + 1), data[: tune + 1], c="maroon", alpha=0.03
                )
                axs[-1][1].plot(range(tune, width), data[tune:], c="C0", alpha=0.045)
                axs[-1][1].plot(
                    [tune, tune],
                    [
                        np.mean(data, 1)[tune] - np.std(data, 1)[tune] * 3,
                        np.mean(data, 1)[tune] + np.std(data, 1)[tune] * 3,
                    ],
                    "--",
                    alpha=0.4,
                    color="k",
                )

            else:
                i95s = np.percentile(data, [2.5, 97.5], axis=1)
                i66s = np.percentile(data, [17, 83], axis=1)
                means = np.mean(data, axis=1)
                medis = np.median(data, axis=1)

                axs[-1][1].fill_between(
                    range(0, tune + 1),
                    *i95s[:, : tune + 1],
                    lw=0,
                    alpha=0.1,
                    color="C1"
                )
                axs[-1][1].fill_between(
                    range(tune, width), *i95s[:, tune:], lw=0, alpha=0.2, color="C1"
                )
                axs[-1][1].fill_between(
                    range(0, tune + 1),
                    *i66s[:, : tune + 1],
                    lw=0,
                    alpha=0.3,
                    color="C1"
                )
                axs[-1][1].fill_between(
                    range(tune, width), *i66s[:, tune:], lw=0, alpha=0.4, color="C1"
                )
                axs[-1][1].plot(range(tune, width), means[tune:], lw=2, c="C0")
                axs[-1][1].plot(
                    range(0, tune + 1), means[: tune + 1], lw=2, c="C0", alpha=0.5
                )

                axs[-1][1].plot(
                    [tune, tune],
                    [
                        np.mean(data, 1)[tune] - np.std(data, 1)[tune] * 3,
                        np.mean(data, 1)[tune] + np.std(data, 1)[tune] * 3,
                    ],
                    "--",
                    alpha=0.4,
                    color="k",
                )

            axs[-1][0].tick_params(
                axis="x",
                direction="out",
                width=1,
                length=3,
                color="0.5",
                labelsize=scale_text(figsize, text_size),
            )
            axs[-1][1].tick_params(
                axis="x",
                direction="out",
                width=1,
                length=3,
                color="0.5",
                labelsize=scale_text(figsize, text_size),
            )
            axs[-1][1].tick_params(
                axis="y",
                width=1,
                length=0,
                color="0.5",
                labelsize=scale_text(figsize, text_size),
            )

            axs[-1][0].set_ylabel("Frequency")
            axs[-1][1].set_ylabel("Sample value")
            axs[-1][0].set_ylim(bottom=0)

            subfigs[-1][i].suptitle(str(varnames[chunk + i]))

        if priors[chunk] is not None:
            axs[-1][0].legend(custom_lines_hist, ["Posterior", "Prior"])
            axs[-1][1].legend(custom_lines_trace, ["Burn-in", "Posterior"])

    return figs, subfigs, axs


def plot_posterior_op(
    trace_values,
    ax,
    bw,
    prior,
    kde_plot=False,
    point_estimate="mean",
    round_to=3,
    alpha_level=0.05,
    ref_val=None,
    rope=None,
    text_size=16,
    display_additinal_info=False,
    **kwargs
):
    """Artist to draw posterior."""

    from .stats import calc_min_interval as hpd

    def format_as_percent(x, round_to=0):
        return "{0:.{1:d}f}%".format(100 * x, round_to)

    def display_ref_val(ref_val):
        less_than_ref_probability = (trace_values < ref_val).mean()
        greater_than_ref_probability = (trace_values >= ref_val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(
            format_as_percent(less_than_ref_probability, 1),
            ref_val,
            format_as_percent(greater_than_ref_probability, 1),
        )
        ax.axvline(ref_val, bottom=0.02, ymax=0.75, color="g", linewidth=4, alpha=0.65)
        ax.text(
            trace_values.mean(),
            plot_height * 0.6,
            ref_in_posterior,
            size=text_size,
            horizontalalignment="center",
        )

    def display_rope(rope):
        ax.plot(
            rope,
            (plot_height * 0.02, plot_height * 0.02),
            linewidth=20,
            color="r",
            alpha=0.75,
        )
        text_props = dict(size=text_size, horizontalalignment="center", color="r")
        ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
        ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ("mode", "mean", "median"):
            raise ValueError("Point Estimate should be in ('mode','mean','median')")
        if point_estimate == "mean":
            point_value = trace_values.mean()
        elif point_estimate == "mode":
            if isinstance(trace_values[0], float):
                density, l, u = fast_kde(trace_values, bw)
                x = np.linspace(l, u, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(trace_values.round(round_to))[0][0]
        elif point_estimate == "median":
            point_value = np.median(trace_values)
        point_text = "{point_estimate}={point_value:.{round_to}f}".format(
            point_estimate=point_estimate, point_value=point_value, round_to=round_to
        )

        ax.text(
            point_value,
            plot_height * 0.8,
            point_text,
            size=text_size,
            horizontalalignment="center",
        )

    def display_hpd():
        sorted_trace_values = np.sort(trace_values)
        hpd_intervals = hpd(sorted_trace_values, alpha=alpha_level)
        ax.plot(
            hpd_intervals,
            (plot_height * 0.02, plot_height * 0.02),
            linewidth=4,
            color="k",
        )
        ax.text(
            hpd_intervals[0],
            plot_height * 0.07,
            hpd_intervals[0].round(round_to),
            size=text_size,
            horizontalalignment="right",
        )
        ax.text(
            hpd_intervals[1],
            plot_height * 0.07,
            hpd_intervals[1].round(round_to),
            size=text_size,
            horizontalalignment="left",
        )
        ax.text(
            (hpd_intervals[0] + hpd_intervals[1]) / 2,
            plot_height * 0.2,
            format_as_percent(1 - alpha_level) + " HPD",
            size=text_size,
            horizontalalignment="center",
        )

    def format_axes():
        ax.yaxis.set_ticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.yaxis.set_ticks_position("none")
        ax.xaxis.set_ticks_position("bottom")
        # ax.tick_params(axis='x', direction='out', width=1, length=3,
        # color='0.5', labelsize=text_size)
        ax.spines["bottom"].set_color("0.5")

    def set_key_if_doesnt_exist(d, key, value):
        if key not in d:
            d[key] = value

    if kde_plot and isinstance(trace_values[0], float):
        kdeplot_op(
            ax, trace_values, bw=bw, prior_alpha=kwargs.pop("alpha", 0.35), **kwargs
        )

    else:
        set_key_if_doesnt_exist(kwargs, "bins", 30)
        set_key_if_doesnt_exist(kwargs, "edgecolor", "w")
        set_key_if_doesnt_exist(kwargs, "align", "right")
        set_key_if_doesnt_exist(kwargs, "color", "#87ceeb")
        ax.hist(trace_values, density=True, **kwargs)

    if prior is not None:
        x = np.linspace(*ax.get_xlim(), 100)
        ax.plot(x, prior.pdf(x), "--", c="C1", alpha=0.8)

    plot_height = ax.get_ylim()[1]

    format_axes()

    if display_additinal_info:
        display_hpd()
        display_point_estimate()
        if ref_val is not None:
            display_ref_val(ref_val)
        if rope is not None:
            display_rope(rope)


def scale_text(figsize, text_size):
    """Scale text to figsize."""

    if text_size is None and figsize is not None:
        if figsize[0] <= 11:
            return 12
        else:
            return figsize[0]
    else:
        return text_size


def posteriorplot(
    trace,
    varnames=None,
    tune=0,
    figsize=None,
    plots_per_fig=4,
    text_size=None,
    ropep=None,
    ref_val=None,
    bw=4.5,
    ax=None,
    **kwargs
):

    axs = []
    figs = []

    if varnames is not None:
        dim = len(varnames)
    else:
        dim = trace.shape[-1]

    for ic in range(0, dim, plots_per_fig):

        if varnames is not None:
            vnames_chunk = varnames[ic : ic + plots_per_fig]
        else:
            vnames_chunk = [None] * min(plots_per_fig, dim - ic)
        trace_chunk = trace[..., ic : ic + plots_per_fig]

        def create_axes_grid(figsize, traces):
            l_trace = len(traces)
            if l_trace == 1:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                n = np.ceil(l_trace / 2.0).astype(int)
                if figsize is None:
                    figsize_loc = (8, len(vnames_chunk) * 1.5)
                fig, ax = plt.subplots(n, 2, figsize=figsize_loc)
                ax = ax.reshape(2 * n)
                if l_trace % 2 == 1:
                    ax[-1].set_axis_off()
                    ax = ax[:-1]
            return fig, ax

        if ax is None:
            fig, ax = create_axes_grid(figsize, vnames_chunk)
        else:
            ax = apx

        var_num = len(vnames_chunk)
        if ref_val is None:
            ref_val = [None] * var_num
        elif np.isscalar(ref_val):
            ref_val = [ref_val for _ in range(var_num)]

        if ropep is None:
            rope = [None] * var_num
        elif np.ndim(ropep) == 1:
            rope = [rope] * var_num

        if len(np.atleast_1d(ax).flatten()) != len(vnames_chunk):
            print("Given axis does not match number of plots")
        for idx, (a, v) in enumerate(zip(np.atleast_1d(ax), vnames_chunk)):
            tr_values = trace_chunk[-tune:, :, idx].flatten()
            plot_posterior_op(
                tr_values,
                ax=a,
                bw=bw,
                prior=None,
                round_to=round_to,
                alpha_level=alpha_level,
                ref_val=ref_val[idx],
                rope=rope[idx],
                text_size=scale_text(figsize, text_size),
                **kwargs
            )
            a.set_title(v, fontsize=scale_text(figsize, text_size))

        plt.tight_layout()

        axs.append(ax)
        figs.append(fig)

    return figs, axs


def sort_nhd(hd):
    """Sort the normalized historical decomposition into negative and positive contributions"""

    hmin = np.zeros_like(hd[0])
    hmax = np.zeros_like(hd[0])
    hmaxt = ()
    hmint = ()

    for h in hd:
        newmax = hmax + np.where(h > 0, h, 0)
        hmaxt += (np.stack((hmax, newmax)),)
        newmin = hmin + np.where(h < 0, h, 0)
        hmint += (np.stack((hmin, newmin)),)
        hmin = newmin
        hmax = newmax

    return hmint, hmaxt
