'''
Created on Feb 4, 2019

@author: Faizan-Uni
'''
import psutil
import numpy as np
from scipy.stats import rankdata

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
