'''
@author: Faizan-Uni-Stuttgart

Nov 9, 2020

6:23:45 PM

'''
import os
import time
import timeit
from pathlib import Path

import matplotlib as mpl
mpl.rc('font', size=14)

import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

plt.ioff()

DEBUG_FLAG = False


def get_corr_mat(phas_spec, mag_spec_norm):

    n_phas, = phas_spec.shape

    corr_mat = np.zeros((n_phas, n_phas), dtype=float)

    for j in range(n_phas):
        for k in range(n_phas):
            if j <= k:
                corr_mat[j, k] = (np.cos(
                    phas_spec[j] - phas_spec[k])) * mag_spec_norm[j]

                if k:
                    corr_mat[j, k] += corr_mat[j, k - 1]
            else:
                corr_mat[j, k] = corr_mat[k, j]

    return corr_mat


def plot_corr_mat(corr_mat_ref, corr_mat_sim, out_fig_name):

    fig = plt.figure(figsize=(15, 7))

    axs = AxesGrid(
        fig,
        111,
        nrows_ncols=(1, 2),
        axes_pad=0.5,
        cbar_mode='single',
        cbar_location='right',
        cbar_pad=0.1)

    vmin = corr_mat_ref.min()
    vmax = corr_mat_ref.max()

    # Reference.
    i = 0
    cb_input = axs[i].imshow(corr_mat_ref[:, :], vmin=vmin, vmax=vmax)

    axs[i].set_title('Reference')

    axs[i].set_xlabel('Phase index')

    axs[i].set_ylabel('Phase index')

    # Simulation.
    i = 1
    axs[i].imshow(corr_mat_sim[:, :], vmin=vmin, vmax=vmax)

    axs[i].set_title('Simulation')

    axs[i].set_xlabel('Phase index')

    # Colorbar.
    # When cbar_mode is 'single', for ax in grid, ax.cax = grid.cbar_axes[0]
    cbar = axs[-1].cax.colorbar(cb_input)  # use this or the next one
#     cbar = axs.cbar_axes[0].colorbar(cb_input)  # use this or the one before
    cbar.set_label_text('correlation')

    plt.suptitle(
        'Phase spectrum cross correlation comparision\n'
        '(cosine of phase difference)')

    plt.savefig(str(out_fig_name), bbox_inches='tight')
    plt.close()
    return


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    sim_dir = Path(r'test_phs_specs_shuff_07_spatial_ppt_2009_2012_rand')

#     main_dir = Path(r'T:\Synchronize_LDs\phsann')
#     os.chdir(main_dir)
#
#     sim_dir = Path(r'test_lag_opt_17')

    out_dir = Path(sim_dir / 'phs_spec_cross_corr')

    out_dir.mkdir(exist_ok=True)

    h5_hdl = h5py.File(sim_dir / r'phsann.h5', 'r')

    # Reference.
    ref_phs_spec = h5_hdl['data_ref_rltzn/_ref_phs_spec'][1:-1]
    ref_mag_spec = h5_hdl['data_ref_rltzn/_ref_mag_spec'][1:-1]

    ref_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

    n_cols = ref_phs_spec.shape[1]

    # Simulations.
    n_sims = h5_hdl['settings'].attrs['_sett_misc_n_rltzns']

    sim_pad_zeros = len(str(n_sims))

    sim_grp = h5_hdl['data_sim_rltzns']

    sim_phs_specs = []
    for i in range(n_sims):
        sim_lab = f'{i:0{sim_pad_zeros}d}'

        sim_phs_spec = sim_grp[f'{sim_lab}/phs_spec'][1:-1]
        sim_phs_specs.append(sim_phs_spec)

        sim_phs_spec = None

    ref_mag_spec_norm = ref_mag_spec / ref_mag_spec.max()
    for i in range(n_cols):
        ref_phs_corr_mat = get_corr_mat(ref_phs_spec[:, i], ref_mag_spec_norm)

        for j in range(n_sims):
            sim_phs_corr_mat_j = get_corr_mat(
                sim_phs_specs[j][:, i], ref_mag_spec_norm)

            out_fig_path = (
                out_dir /
                (f'ss__phs_corr_mat_{ref_labels[i]}_{j:0{sim_pad_zeros}d}.png'))

            plot_corr_mat(ref_phs_corr_mat, sim_phs_corr_mat_j, out_fig_path)

    return


if __name__ == '__main__':

    _save_log_ = False
    if _save_log_:
        from datetime import datetime
        from std_logger import StdFileLoggerCtrl

        # save all console activity to out_log_file
        out_log_file = os.path.join(
            r'P:\Synchronize\python_script_logs\\%s_log_%s.log' % (
            os.path.basename(__file__),
            datetime.now().strftime('%Y%m%d%H%M%S')))

        log_link = StdFileLoggerCtrl(out_log_file)

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
            import pdb
            pdb.post_mortem()

    else:
        main()

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))

    if _save_log_:
        log_link.stop()
