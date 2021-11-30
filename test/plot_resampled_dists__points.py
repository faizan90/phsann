'''
@author: Faizan-Uni-Stuttgart

Nov 29, 2021

3:09:16 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt; plt.ioff()

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\phd_sims__dual_420_427_asymm12_with_ft_06_phsrand')

    os.chdir(main_dir)

    data_dir = Path(r'resampled_dists__points_time')

    # out_fig_pref = 'RTsum'
    # ref_data_patt = f'ref_data__{out_fig_pref}.csv'
    # sim_data_patt = f'sim_data_*__{out_fig_pref}.csv'

    out_fig_pref = 'RTsum__WS14D_RTsum'
    ref_data_patt = f'ref_data__{out_fig_pref}.csv'
    sim_data_patt = f'sim_data_*__{out_fig_pref}.csv'

    fig_x_label = '14 days station rolling sum [-]'
    fig_y_label = '1 - F(x) [-]'

    out_dir = Path(r'resampled_dists__points_time_plots')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    read_ref_flag = False
    for ref_data_file in data_dir.glob(ref_data_patt):
        read_ref_flag = True

    assert read_ref_flag, 'Didn\'t find the reference file!'

    ref_data_ser = pd.read_csv(
        ref_data_file, sep=';', index_col=0, squeeze=True)

    assert isinstance(ref_data_ser, pd.Series)

    sim_data_sers = []
    for sim_data_file in data_dir.glob(sim_data_patt):
        sim_data_ser = pd.read_csv(
            sim_data_file, sep=';', index_col=0, squeeze=True)

        assert isinstance(sim_data_ser, pd.Series)

        sim_data_sers.append(sim_data_ser)

    assert sim_data_sers, 'Didn\'t find the simulation file(s)!'

    plt.figure(figsize=(7, 7))
    leg_flag = True
    for sim_data_ser in sim_data_sers:
        if leg_flag:
            label = 'sim'
            leg_flag = False

        else:
            label = None

        sim_ser = sim_data_ser.sort_values()
        sim_probs = sim_ser.rank().values / (sim_ser.shape[0] + 1.0)

        plt.semilogy(
            sim_ser.values,
            1 - sim_probs,
            c='k',
            alpha=0.4,
            lw=1.5,
            label=label)

    ref_ser = ref_data_ser.sort_values()
    ref_probs = ref_ser.rank().values / (ref_ser.shape[0] + 1.0)

    plt.semilogy(
        ref_ser.values,
        1 - ref_probs,
        c='r',
        alpha=0.8,
        lw=2,
        label='ref')

    plt.grid(which='both')
    plt.gca().set_axisbelow(True)

    plt.legend()

    plt.xlabel(fig_x_label)
    plt.ylabel(fig_y_label)

    plt.savefig(
        out_dir / f'{out_fig_pref}.png',
        dpi=150,
        bbox_inches='tight')

    plt.clf()

    plt.close()
    return


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    #==========================================================================
    # When in post_mortem:
    # 1. "where" to show the stack,
    # 2. "up" move the stack up to an older frame,
    # 3. "down" move the stack down to a newer frame, and
    # 4. "interact" start an interactive interpreter.
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
