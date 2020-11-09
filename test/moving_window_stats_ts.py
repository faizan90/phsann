'''
@author: Faizan-Uni-Stuttgart

Nov 5, 2020

6:43:43 PM

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt

plt.ioff()

DEBUG_FLAG = False


def get_mw_mean_med_vrc(data, ws):

    mean_arr = np.zeros(data.shape[0] - ws)
    med_arr = mean_arr.copy()
    vrc_arr = mean_arr.copy()
    mins_arr = mean_arr.copy()
    maxs_arr = mean_arr.copy()

    ws_xcrds = []
    for i in range(data.shape[0] - ws):
        mean_arr[i] = data[i:i + ws].mean()
        med_arr[i] = np.median(data[i:i + ws])
        vrc_arr[i] = data[i:i + ws].var()
        mins_arr[i] = data[i:i + ws].min()
        maxs_arr[i] = data[i:i + ws].max()

        ws_xcrds.append(i + int(0.5 * ws))

    ws_xcrds = np.array(ws_xcrds)
    return (ws_xcrds, mean_arr, med_arr, vrc_arr, mins_arr, maxs_arr)


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    sim_dir = Path(r'test_phs_specs_shuff_03_spatial_dis_2009_2012_rand')

    windows_sizes = [30, 60, 365]

    fig_size = (12, 7)

    out_dir = Path(sim_dir / 'mw_stat_figs_ts')

    out_dir.mkdir(exist_ok=True)

    h5_hdl = h5py.File(sim_dir / r'phsann.h5', 'r')

    # Reference realization.
    ref_data = h5_hdl['data_ref/_data_ref_rltzn'][...]

    ref_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

    n_cols = ref_data.shape[1]

    # Simulations
    n_sims = h5_hdl['settings'].attrs['_sett_misc_n_rltzns']

    sim_pad_zeros = len(str(n_sims))

    sim_grp = h5_hdl['data_sim_rltzns']

    sim_datas = []
    for i in range(n_sims):
        sim_lab = f'{i:0{sim_pad_zeros}d}'

        sim_data = sim_grp[f'{sim_lab}/data'][...]

        sim_datas.append(sim_data)

        sim_data = None

    h5_hdl.close()

    for ws in windows_sizes:
        for i in range(n_cols):
            print(ws, i)

            x_crds, mean_arr, med_arr, vrc_arr, min_arr, max_arr = (
                get_mw_mean_med_vrc(ref_data[:, i], ws))

            means_fig = plt.figure(figsize=fig_size)
            meds_fig = plt.figure(figsize=fig_size)
            vrcs_fig = plt.figure(figsize=fig_size)
            mins_fig = plt.figure(figsize=fig_size)
            maxs_fig = plt.figure(figsize=fig_size)

            label = 'ref'

            plt.figure(means_fig.number)
            plt.plot(x_crds, mean_arr, label=label, alpha=0.7, lw=2, c='red')

            plt.figure(meds_fig.number)
            plt.plot(x_crds, med_arr, label=label, alpha=0.7, lw=2, c='red')

            plt.figure(vrcs_fig.number)
            plt.plot(x_crds, vrc_arr, label=label, alpha=0.7, lw=2, c='red')

            plt.figure(mins_fig.number)
            plt.plot(x_crds, min_arr, label=label, alpha=0.7, lw=2, c='red')

            plt.figure(maxs_fig.number)
            plt.plot(x_crds, max_arr, label=label, alpha=0.7, lw=2, c='red')

            for j in range(n_sims):
                x_crds, mean_arr, med_arr, vrc_arr, min_arr, max_arr = (
                    get_mw_mean_med_vrc(sim_datas[j][:, i], ws))

                if not j:
                    label = 'sim'

                else:
                    label = None

                plt.figure(means_fig.number)
                plt.plot(x_crds, mean_arr, label=label, alpha=0.3, lw=1, c='gray')

                plt.figure(meds_fig.number)
                plt.plot(x_crds, med_arr, label=label, alpha=0.3, lw=1, c='gray')

                plt.figure(vrcs_fig.number)
                plt.plot(x_crds, vrc_arr, label=label, alpha=0.3, lw=1, c='gray')

                plt.figure(mins_fig.number)
                plt.plot(x_crds, min_arr, label=label, alpha=0.3, lw=1, c='gray')

                plt.figure(maxs_fig.number)
                plt.plot(x_crds, max_arr, label=label, alpha=0.3, lw=1, c='gray')

            plt.figure(means_fig.number)
            plt.xlabel('Step')
            plt.ylabel(f'Moving window mean ({ws} steps)')
            plt.grid()
            plt.legend()
            out_fig_name = f'mw_mean_{ref_labels[i]}_{ws}.png'
            plt.savefig(
                str(out_dir / out_fig_name), bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(meds_fig.number)
            plt.xlabel('Step')
            plt.ylabel(f'Moving window median ({ws} steps)')
            plt.grid()
            plt.legend()
            out_fig_name = f'mw_med_{ref_labels[i]}_{ws}.png'
            plt.savefig(
                str(out_dir / out_fig_name), bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(vrcs_fig.number)
            plt.xlabel('Step')
            plt.ylabel(f'Moving window variance ({ws} steps)')
            plt.grid()
            plt.legend()
            out_fig_name = f'mw_vrc_{ref_labels[i]}_{ws}.png'
            plt.savefig(
                str(out_dir / out_fig_name), bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(mins_fig.number)
            plt.xlabel('Step')
            plt.ylabel(f'Moving window minima ({ws} steps)')
            plt.grid()
            plt.legend()
            out_fig_name = f'mw_min_{ref_labels[i]}_{ws}.png'
            plt.savefig(
                str(out_dir / out_fig_name), bbox_inches='tight', dpi=300)
            plt.close()

            plt.figure(maxs_fig.number)
            plt.xlabel('Step')
            plt.ylabel(f'Moving window maxima ({ws} steps)')
            plt.grid()
            plt.legend()
            out_fig_name = f'mw_max_{ref_labels[i]}_{ws}.png'
            plt.savefig(
                str(out_dir / out_fig_name), bbox_inches='tight', dpi=300)
            plt.close()

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
