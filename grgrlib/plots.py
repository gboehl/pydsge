#!/bin/python2
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def pplot(X, labels=None, yscale=None, title='', style='-', savepath=None, Y=None):

    plt_no      = X.shape[1] // 4 + bool(X.shape[1]%4)

    if yscale is None:
        yscale  = np.arange(X.shape[0])

    if labels is None:
        labels  = np.arange(X.shape[1]) + 1

    axs     = []
    for i in range(plt_no):

        ax  = plt.subplots(2,2)[1].flatten()

        for j in range(4):

            if 4*i+j >= X.shape[1]:
                ax[j].set_visible(False)

            else:
                if X.shape[1] > 4*i+j:
                    ax[j].plot(yscale, X[:,4*i+j], style, lw=2)

                if Y is not None:
                    if Y.shape[1] > 4*i+j:
                        ax[j].plot(yscale, Y[:,4*i+j], style, lw=2)

                ax[j].tick_params(axis='both', which='both', top=False, right=False, labelsize=12)
                ax[j].spines['top'].set_visible(False)
                ax[j].spines['right'].set_visible(False)
                ax[j].set_xlabel(labels[4*i+j], fontsize=14)

        if title:
            plt.suptitle('%s %s' %(title,i+1), fontsize=16)

        plt.tight_layout()

        if savepath is not None:
            plt.savefig(savepath+title+str(i+1)+'.pdf')

        axs.append(ax)

    return axs


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
        std_x = 0.
    grid, _ = np.histogram(x, bins=nx)

    scotts_factor = n ** (-0.2)
    kern_nx = int(scotts_factor * 2 * np.pi * std_x)
    kernel = gaussian(kern_nx, scotts_factor * std_x)

    npad = min(nx, 2 * kern_nx)
    grid = np.concatenate([grid[npad: 0: -1], grid, grid[nx: nx - npad: -1]])
    density = convolve(grid, kernel, mode='same')[npad: npad + nx]

    norm_factor = n * dx * (2 * np.pi * std_x ** 2 * scotts_factor ** 2) ** 0.5

    density = density / norm_factor

    return density, xmin, xmax

def kdeplot_op(ax, data, bw, prior=None, prior_alpha=1, prior_style='--'):
    """Get a list of density and likelihood plots, if a prior is provided."""
    ls = []
    pls = []
    errored = []
    try:
        density, l, u = fast_kde(data, bw)
        x = np.linspace(l, u, len(density))
        if prior is not None:
            p = prior.logpdf(x)
            pls.append(ax.plot(x, np.exp(p),
                               alpha=prior_alpha, ls=prior_style))

        ls.append(ax.plot(x, density))
    except ValueError:
        errored.append(str(i))

    if errored:
        ax.text(.27, .47, 'WARNING: KDE plot failed for: ' + ','.join(errored),
                bbox={'facecolor': 'red', 'alpha': 0.5, 'pad': 10},
                style='italic')

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
        _, ax = plt.subplots(*default_shape, **default_kwargs)
    elif ax.shape != default_shape:
        raise ValueError('Subplots with shape %r required' % (default_shape,))
    return ax

def traceplot(trace, varnames, tune, figsize=None,
              combined=False, grid=False, alpha=0.35, priors=None,
              prior_alpha=1, prior_style='--', bw=4.5, ax=None):
    ## stolen from pymc3 with kisses

    if figsize is None:
        figsize = (12, len(varnames) * 2)

    ax = get_axis(ax, len(varnames), 2, squeeze=False, figsize=figsize)

    for i, v in enumerate(varnames):
        if priors is not None:
            prior = priors[i]
        else:
            prior = None
        d = trace[:,:,i]
        d_stream = d.swapaxes(0,1)
        d   = d_stream
        x0 = 0
        width = len(d_stream)
        artists = kdeplot_op(ax[i, 0], d[tune:], bw, prior, prior_alpha, prior_style)[0]
        colors = [a[0].get_color() for a in artists]
        ax[i, 0].set_title(str(v))
        ax[i, 0].grid(grid)
        ax[i, 1].set_title(str(v))
        ax[i, 1].plot(range(x0, tune), d_stream[:tune], alpha=alpha-.1)
        ax[i, 1].plot(range(tune, x0 + width), d_stream[tune:], alpha=alpha)
        ax[i, 1].plot([tune,tune], [ np.mean(d_stream, 1)[tune] - np.std(d_stream, 1)[tune]*3, 
                                    np.mean(d_stream, 1)[tune] + np.std(d_stream, 1)[tune]*3],
                      '--', alpha=.5, color='k')

        ax[i, 0].set_ylabel("Frequency")
        ax[i, 1].set_ylabel("Sample value")
        ax[i, 0].set_ylim(ymin=0)
    plt.tight_layout()
    return ax


def plot_posterior_op(trace_values, ax, bw, kde_plot, point_estimate, round_to,
                      alpha_level, ref_val, rope, text_size=16, **kwargs):
    """Artist to draw posterior."""

    from .stats import calc_min_interval as hpd

    def format_as_percent(x, round_to=0):
        return '{0:.{1:d}f}%'.format(100 * x, round_to)

    def display_ref_val(ref_val):
        less_than_ref_probability = (trace_values < ref_val).mean()
        greater_than_ref_probability = (trace_values >= ref_val).mean()
        ref_in_posterior = "{} <{:g}< {}".format(
            format_as_percent(less_than_ref_probability, 1),
            ref_val,
            format_as_percent(greater_than_ref_probability, 1))
        ax.axvline(ref_val, ymin=0.02, ymax=.75, color='g',
                   linewidth=4, alpha=0.65)
        ax.text(trace_values.mean(), plot_height * 0.6, ref_in_posterior,
                size=text_size, horizontalalignment='center')

    def display_rope(rope):
        ax.plot(rope, (plot_height * 0.02, plot_height * 0.02),
                linewidth=20, color='r', alpha=0.75)
        text_props = dict(size=text_size, horizontalalignment='center', color='r')
        ax.text(rope[0], plot_height * 0.14, rope[0], **text_props)
        ax.text(rope[1], plot_height * 0.14, rope[1], **text_props)

    def display_point_estimate():
        if not point_estimate:
            return
        if point_estimate not in ('mode', 'mean', 'median'):
            raise ValueError(
                "Point Estimate should be in ('mode','mean','median')")
        if point_estimate == 'mean':
            point_value = trace_values.mean()
        elif point_estimate == 'mode':
            if isinstance(trace_values[0], float):
                density, l, u = fast_kde(trace_values, bw)
                x = np.linspace(l, u, len(density))
                point_value = x[np.argmax(density)]
            else:
                point_value = mode(trace_values.round(round_to))[0][0]
        elif point_estimate == 'median':
            point_value = np.median(trace_values)
        point_text = '{point_estimate}={point_value:.{round_to}f}'.format(point_estimate=point_estimate,
                                                                          point_value=point_value, round_to=round_to)

        ax.text(point_value, plot_height * 0.8, point_text,
                size=text_size, horizontalalignment='center')

    def display_hpd():
        sorted_trace_values     = np.sort(trace_values)
        hpd_intervals = hpd(sorted_trace_values, alpha=alpha_level)
        ax.plot(hpd_intervals, (plot_height * 0.02,
                                plot_height * 0.02), linewidth=4, color='k')
        ax.text(hpd_intervals[0], plot_height * 0.07,
                hpd_intervals[0].round(round_to),
                size=text_size, horizontalalignment='right')
        ax.text(hpd_intervals[1], plot_height * 0.07,
                hpd_intervals[1].round(round_to),
                size=text_size, horizontalalignment='left')
        ax.text((hpd_intervals[0] + hpd_intervals[1]) / 2, plot_height * 0.2,
                format_as_percent(1 - alpha_level) + ' HPD',
                size=text_size, horizontalalignment='center')

    def format_axes():
        ax.yaxis.set_ticklabels([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.yaxis.set_ticks_position('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='x', direction='out', width=1, length=3,
                       color='0.5', labelsize=text_size)
        ax.spines['bottom'].set_color('0.5')

    def set_key_if_doesnt_exist(d, key, value):
        if key not in d:
            d[key] = value

    if kde_plot and isinstance(trace_values[0], float):
        kdeplot(trace_values, alpha=kwargs.pop('alpha', 0.35), bw=bw, ax=ax, **kwargs)

    else:
        set_key_if_doesnt_exist(kwargs, 'bins', 30)
        set_key_if_doesnt_exist(kwargs, 'edgecolor', 'w')
        set_key_if_doesnt_exist(kwargs, 'align', 'right')
        set_key_if_doesnt_exist(kwargs, 'color', '#87ceeb')
        ax.hist(trace_values, **kwargs)

    plot_height = ax.get_ylim()[1]

    format_axes()
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


def posteriorplot(trace, varnames=None, tune=0, figsize=None, text_size=None,
                   alpha_level=0.05, round_to=3, point_estimate='mean', rope=None,
                   ref_val=None, kde_plot=False, bw=4.5, ax=None, **kwargs):

    def create_axes_grid(figsize, traces):
        l_trace = len(traces)
        if l_trace == 1:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            n = np.ceil(l_trace / 2.0).astype(int)
            if figsize is None:
                figsize = (12, n * 2.5)
            fig, ax = plt.subplots(n, 2, figsize=figsize)
            ax = ax.reshape(2 * n)
            if l_trace % 2 == 1:
                ax[-1].set_axis_off()
                ax = ax[:-1]
        return fig, ax

    if ax is None:
        fig, ax = create_axes_grid(figsize, varnames)

    var_num = len(varnames)
    if ref_val is None:
        ref_val = [None] * var_num
    elif np.isscalar(ref_val):
        ref_val = [ref_val for _ in range(var_num)]

    if rope is None:
        rope = [None] * var_num
    elif np.ndim(rope) == 1:
        rope = [rope] * var_num

    # for idx, (a, v) in enumerate(zip(np.atleast_1d(ax), varnames)):
    for idx, (a, v) in enumerate(zip(ax.flatten(), varnames)):
        tr_values = trace[:,tune:,idx].flatten()
        plot_posterior_op(tr_values, ax=a, bw=bw, kde_plot=kde_plot,
                          point_estimate=point_estimate, round_to=round_to,
                          alpha_level=alpha_level, ref_val=ref_val[idx],
                          rope=rope[idx], text_size=scale_text(figsize, text_size), **kwargs)
        a.set_title(v, fontsize=scale_text(figsize, text_size))

    plt.tight_layout()
    return ax
