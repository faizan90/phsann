'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''
import psutil
import numpy as np
from scipy.stats import rankdata

from .cyth import (
    fill_bin_idxs_ts,
    fill_bin_dens_1d,
    fill_bin_dens_2d,
    fill_etpy_lcl_ts)

print_line_str = 79 * '#'


def print_sl():

    print(2 * '\n', print_line_str, sep='')
    return


def print_el():

    print(print_line_str)
    return


def get_n_cpus():

    phy_cores = psutil.cpu_count(logical=False)
    log_cores = psutil.cpu_count()

    if phy_cores < log_cores:
        n_cpus = phy_cores

    else:
        n_cpus = log_cores - 1

    n_cpus = max(n_cpus, 1)

    return n_cpus


def ret_mp_idxs(n_vals, n_cpus):

    assert n_vals > 0

    idxs = np.linspace(
        0, n_vals, min(n_vals + 1, n_cpus + 1), endpoint=True, dtype=np.int64)

    idxs = np.unique(idxs)

    assert idxs.shape[0]

    if idxs.shape[0] == 1:
        idxs = np.concatenate((np.array([0]), idxs))

    assert (idxs[0] == 0) & (idxs[-1] == n_vals), idxs
    return idxs


def roll_real_2arrs(arr1, arr2, lag, rerank_flag=False):

    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)

    assert arr1.ndim == 1
    assert arr2.ndim == 1

    assert arr1.size == arr2.size

    assert isinstance(lag, (int, np.int64))
    assert abs(lag) < arr1.size

    if lag > 0:
        # arr2 is shifted ahead
        arr1 = arr1[:-lag].copy()
        arr2 = arr2[+lag:].copy()

    elif lag < 0:
        # arr1 is shifted ahead
        arr1 = arr1[+lag:].copy()
        arr2 = arr2[:-lag].copy()

    else:
        pass

    assert arr1.size == arr2.size

    if rerank_flag:
#         assert np.all(arr1 > 0) and np.all(arr2 > 0)
#         assert np.all(arr1 < 1) and np.all(arr2 < 1)

        arr1 = rankdata(arr1) / (arr1.size + 1.0)
        arr2 = rankdata(arr2) / (arr2.size + 1.0)

    return arr1, arr2

# def get_binned_ts(probs, n_bins):
#
#     assert np.all(probs > 0) and np.all(probs < 1)
#
#     assert n_bins > 1
#     assert n_bins < probs.size
#
#     bin_idxs_ts = (probs * n_bins).astype(int)
#
#     assert np.all(bin_idxs_ts >= 0) and np.all(bin_idxs_ts < n_bins)
#
#     return bin_idxs_ts
#
#
# def get_binned_dens_ftn_1d(bin_idxs_ts, n_bins):
#
#     bin_freqs = np.unique(bin_idxs_ts, return_counts=True)[1]
#     bin_dens = bin_freqs * (1 / n_bins)
#
#     return bin_dens
#
#
# def get_binned_dens_ftn_2d(probs_1, probs_2, n_bins):
#
#     bins = np.linspace(0.0, 1.0, n_bins + 1)
#
#     bin_freqs_12 = np.histogram2d(probs_1, probs_2, bins=bins)[0]
#
#     bin_dens_12 = bin_freqs_12 * ((1 / n_bins) ** 2)
#
#     return bin_dens_12
#
#
# def get_local_entropy_ts(probs_1, probs_2, n_bins):
#
#     bin_idxs_ts_1 = get_binned_ts(probs_1, n_bins)
#     bin_idxs_ts_2 = get_binned_ts(probs_2, n_bins)
#
#     bin_dens_1 = get_binned_dens_ftn_1d(bin_idxs_ts_1, n_bins)
#     bin_dens_2 = get_binned_dens_ftn_1d(bin_idxs_ts_2, n_bins)
#
#     bin_dens_12 = get_binned_dens_ftn_2d(probs_1, probs_2, n_bins)
#
# #     etpy_local = np.empty_like(bin_idxs_ts_1, dtype=float)
# #     for i in range(bin_idxs_ts_1.shape[0]):
# #
# #         dens = bin_dens_12[bin_idxs_ts_1[i], bin_idxs_ts_2[i]]
# #
# #         if not dens:
# #             etpy_local[i] = 0
# #
# #         else:
# #             prod = bin_dens_1[bin_idxs_ts_1[i]] * bin_dens_2[bin_idxs_ts_2[i]]
# #             etpy_local[i] = (dens * np.log(dens / prod))
#
#     # Mutual information.
#     dens = bin_dens_12[bin_idxs_ts_1, bin_idxs_ts_2]
#     prods = bin_dens_1[bin_idxs_ts_1] * bin_dens_2[bin_idxs_ts_2]
#
#     dens_idxs = dens.astype(bool)
#
#     etpy_local = np.zeros_like(bin_idxs_ts_1, dtype=float)
#
#     etpy_local[dens_idxs] = -dens[dens_idxs] * np.log(
#         dens[dens_idxs] / prods[dens_idxs])
#
# #     # Relative entropy.
# #     etpy_local = bin_dens_1[bin_idxs_ts_1] * np.log(
# #         bin_dens_1[bin_idxs_ts_1] / bin_dens_2[bin_idxs_ts_2])
#
#     # Conditional entropy.
# #     dens = bin_dens_12[bin_idxs_ts_1, bin_idxs_ts_2]
# #     prods = bin_dens_1[bin_idxs_ts_1]  # * bin_dens_2[bin_idxs_ts_2]
# #
# #     dens_idxs = dens.astype(bool)
# #
# #     etpy_local = np.zeros_like(bin_idxs_ts_1, dtype=float)
# #
# #     etpy_local[dens_idxs] = dens[dens_idxs] * np.log(
# #         dens[dens_idxs] / prods[dens_idxs])
#
#     return etpy_local


def get_local_entropy_ts_cy(probs_x, probs_y, n_bins):

    bins_ts_x = np.empty_like(probs_x, dtype=np.uint32)
    bins_ts_y = np.empty_like(probs_y, dtype=np.uint32)

    bins_dens_x = np.empty(n_bins, dtype=float)
    bins_dens_y = np.empty(n_bins, dtype=float)

    bins_dens_xy = np.empty((n_bins, n_bins), dtype=float)

    lcl_etpy_ts = np.empty_like(probs_x, dtype=float)

    fill_bin_idxs_ts(probs_x, bins_ts_x, n_bins)
    fill_bin_idxs_ts(probs_y, bins_ts_y, n_bins)

    fill_bin_dens_1d(bins_ts_x, bins_dens_x)
    fill_bin_dens_1d(bins_ts_y, bins_dens_y)

    fill_bin_dens_2d(bins_ts_x, bins_ts_y, bins_dens_xy)

    fill_etpy_lcl_ts(
        bins_ts_x,
        bins_ts_y,
        bins_dens_x,
        bins_dens_y,
        lcl_etpy_ts,
        bins_dens_xy)

    return lcl_etpy_ts
