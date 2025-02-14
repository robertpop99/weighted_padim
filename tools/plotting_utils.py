# This code was adapted from: https://github.com/openvinotoolkit/anomalib

# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axis import Axis
from matplotlib.figure import Figure
from torch import Tensor


def plot_figure(
        x_vals: Tensor,
        y_vals: Tensor,
        auc: Tensor,
        xlim: Tuple[float, float],
        ylim: Tuple[float, float],
        xlabel: str,
        ylabel: str,
        loc: str,
        title: str,
        sample_points: int = 1000,
) -> Tuple[Figure, Axis]:
    """Generate a simple, ROC-style plot, where x_vals is plotted against y_vals.

    Note that a subsampling is applied if > sample_points are present in x/y, as matplotlib plotting draws
    every single plot which takes very long, especially for high-resolution segmentations.

    Args:
        x_vals (Tensor): x values to plot
        y_vals (Tensor): y values to plot
        auc (Tensor): normalized area under the curve spanned by x_vals, y_vals
        xlim (Tuple[float, float]): displayed range for x-axis
        ylim (Tuple[float, float]): displayed range for y-axis
        xlabel (str): label of x axis
        ylabel (str): label of y axis
        loc (str): string-based legend location, for details see
            https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html
        title (str): title of the plot
        sample_points (int): number of sampling points to subsample x_vals/y_vals with

    Returns:
        Tuple[Figure, Axis]: Figure and the contained Axis
    """
    fig, axis = plt.subplots()

    x_vals = x_vals.detach().cpu()
    y_vals = y_vals.detach().cpu()

    if sample_points < x_vals.size(0):
        possible_idx = range(x_vals.size(0))
        interval = len(possible_idx) // sample_points

        idx = [0]  # make sure to start at first point
        idx.extend(possible_idx[::interval])
        idx.append(possible_idx[-1])  # also include last point

        idx = torch.tensor(
            idx,
            device=x_vals.device,
        )
        x_vals = torch.index_select(x_vals, 0, idx)
        y_vals = torch.index_select(y_vals, 0, idx)

    axis.plot(
        x_vals,
        y_vals,
        color="darkorange",
        figure=fig,
        lw=2,
        label=f"AUC: {auc.detach().cpu():0.2f}",
    )

    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.legend(loc=loc)
    axis.set_title(title)
    return fig, axis


def plot_insar(img, vmin=None, vmax=None) -> Tuple[Figure, Axis]:
    fig, ax = plt.subplots()
    pos = ax.imshow(img, cmap='jet')
    # if vmin is None or vmax is None:
    #     pos = ax.imshow(img, cmap='jet')
    # else:
    #     pos = ax.imshow(img, cmap='jet', vmin=vmin, vmax=vmax)
    cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
    fig.colorbar(pos, cax=cax)
    return fig, ax


def gallery(array, ncols):
    if len(array.shape) == 3:
        nindex, height, width = array.shape
    else:
        nindex, _, height, width = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1, 2)
              .reshape(height * nrows, width * ncols))
    return result


def save_insar(img, label, plot_name, ncols):
    dataset = 'unknown'
    vmin = -np.pi
    vmax = np.pi
    # for set in datasets_info:
    #     if set in label:
    #         dataset = set
    # if 'diff_pha' in label:
    #     pass
    # elif 'unw.' in label:
    #     vmin = datasets_info[dataset]['unw']['mean'] - 3 * datasets_info[dataset]['unw']['std']
    #     vmax = datasets_info[dataset]['unw']['mean'] + 3 * datasets_info[dataset]['unw']['std']
    # elif 'unw_GACOS' in label:
    #     vmin = datasets_info[dataset]['unw_GACOS']['mean'] - 3 * datasets_info[dataset]['unw_GACOS']['std']
    #     vmax = datasets_info[dataset]['unw_GACOS']['mean'] + 3 * datasets_info[dataset]['unw_GACOS']['std']

    table = gallery(img, ncols)
    fig, _ = plot_insar(table, vmin=vmin, vmax=vmax)
    fig.savefig(plot_name, bbox_inches='tight')
    plt.close(fig)


def plot_img_to_file(img, file_name, vmin=None, vmax=None):
    if torch.is_tensor(img):
        if img.is_cuda:
            img = img.detach().cpu()
        img = img.numpy()

    if len(img.shape) > 2:
        if len(img.shape) == 3:
            if img.shape[0] == 1:
                img = img[0, :, :]
            else:
                print(f"wrong shape {img.shape}")
        else:
            print(f"wrong shape {img.shape}")

    plt.imshow(img, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()
