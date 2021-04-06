'''
@author: Faizan-Uni-Stuttgart

Mar 12, 2021

11:29:48 AM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd
# import pyinform as pim
import matplotlib.pyplot as plt

from phsann.misc import (
    roll_real_2arrs,
#     get_local_entropy_ts,
    get_local_entropy_ts_cy,
#     get_binned_ts,
#     get_binned_dens_ftn_1d,
#     get_binned_dens_ftn_2d,
    )

from phsann.cyth import (
    fill_bin_idxs_ts,
    fill_bin_dens_1d,
    fill_bin_dens_2d,
    fill_etpy_lcl_ts)

plt.ioff()

DEBUG_FLAG = True


def get_cyth_lcl_etpy(
        probs_x,
        probs_y,
        n_bins,
        bins_ts_x,
        bins_ts_y,
        bins_dens_x,
        bins_dens_y,
        bins_dens_xy):

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


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    data_file = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\neckar_norm_cop_infill_discharge_1961_2015_20190118.csv')

    beg_time = '1999-01-01'
    end_time = '2000-12-31'

    col = '420'

    max_lags = 3

    n_sims = 1000

    n_bins = 50

    data = pd.read_csv(
        data_file, sep=';', index_col=0).loc[beg_time:end_time, col].values

    if data.size % 2:
        data = data[:-1]

    assert np.all(np.isfinite(data))

    probs_1, probs_2 = roll_real_2arrs(data, data, max_lags, True)

#     # Python.
#     beg_time = timeit.default_timer()
#
#     for _ in range(n_sims):
#         etpy_lcl = get_local_entropy_ts(probs_1, probs_2, n_bins)
#
#     end_time = timeit.default_timer()
#
#     print(f'Python time: {end_time - beg_time:0.2f}')
#     print(f'Python mean time: {(end_time - beg_time) / n_sims:0.3E}')

#     # Pyinform
#     beg_time = timeit.default_timer()
#
#     for _ in range(n_sims):
#         ai = pim.mutual_info(
#             (probs_1 * n_bins
#             ).astype(int),
#             (probs_2 * n_bins
#             ).astype(int), True)
#
#     end_time = timeit.default_timer()
#
#     print(f'Pyinform time: {end_time - beg_time:0.2f}')
#     print(f'Pyinform mean time: {(end_time - beg_time) / n_sims:0.3E}')

#     # Python optimized.
#     beg_time = timeit.default_timer()
#
#     bin_idxs_ts_1 = get_binned_ts(probs_1, n_bins)
#     bin_idxs_ts_2 = get_binned_ts(probs_2, n_bins)
#
#     bin_dens_1 = get_binned_dens_ftn_1d(bin_idxs_ts_1, n_bins)
#     bin_dens_2 = get_binned_dens_ftn_1d(bin_idxs_ts_2, n_bins)
#
#     for _ in range(n_sims):
#
#         bin_idxs_ts_1 = get_binned_ts(probs_1, n_bins)
#         bin_idxs_ts_2 = get_binned_ts(probs_2, n_bins)
#
#         bin_dens_12 = get_binned_dens_ftn_2d(probs_1, probs_2, n_bins)
#
#         prods = bin_dens_1[bin_idxs_ts_1] * bin_dens_2[bin_idxs_ts_2]
#
#         dens = bin_dens_12[bin_idxs_ts_1, bin_idxs_ts_2]
#
#         dens_idxs = dens.astype(bool)
#
#         etpy_local = np.zeros_like(bin_idxs_ts_1, dtype=float)
#
#         etpy_local[dens_idxs] = dens[dens_idxs] * np.log(
#             dens[dens_idxs] / prods[dens_idxs])
#
#     end_time = timeit.default_timer()
#
#     print(f'Python optimized time: {end_time - beg_time:0.2f}')
#     print(f'Python optimized mean time: {(end_time - beg_time) / n_sims:0.3E}')

    # Cython.
    beg_time = timeit.default_timer()

    for _ in range(n_sims):
        etpy_lcl = get_local_entropy_ts_cy(probs_1, probs_2, n_bins)

    end_time = timeit.default_timer()

    print(f'Cython time: {end_time - beg_time:0.2f}')
    print(f'Cython mean time: {(end_time - beg_time) / n_sims:0.3E}')

#     print(ai)
#     print(etpy_local)
#
    ax1 = plt.subplots(1, 1, figsize=(50, 7))[1]

    ax1.plot(data, label='data', c='r', alpha=0.7)

    ax1.legend(loc=1)
    ax1.grid()
    ax1.set_axisbelow(True)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Discharge')

    ax2 = ax1.twinx()
    ax2.plot(etpy_lcl, label='etpy_cy', c='k', alpha=0.7)
#     ax2.plot(ai, label='etpy_pim', c='b', alpha=0.7)

    ax2.legend(loc=2)
    ax2.set_ylabel('Entropy')

    plt.savefig(r'P:/Downloads/etpy_cy_plus.png', dpi=300, bbox_inches='tight')
    plt.close()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack
    # 2. "up" move the stack up to an older frame
    # 3. "down" move the stack down to a newer frame
    # 4. "interact" start an interactive interpreter
    #==========================================================================

    if DEBUG_FLAG:
        try:
            main()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

            import pdb
            pdb.post_mortem()
    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
