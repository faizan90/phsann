'''
@author: Faizan-Uni-Stuttgart

26 May 2020

14:41:57

'''
import os
import time
import timeit
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')

    main_dir /= r'test_phsrand_ppt_hourly_02'

    os.chdir(main_dir)

    h5_file = Path(r"phsann.h5")

    # If you want the time series to have an index.
    # Should have the correct length.

    # time_idx = None
    # time_idx = pd.date_range('1991-01-01', '1995-12-31', freq='D')

    time_idx = pd.date_range(
        '2010-01-01 00:00:00', '2014-12-31 23:00:00', freq='H')

    out_dir = Path(h5_file.parents[0].stem) / 'data_extracted'
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    h5_hdl = h5py.File(h5_file, 'r')

    # Reference realization.
    ref_data = h5_hdl['data_ref/data_ref_rltzn'][...]

    ref_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

    if time_idx is None:
        np.savetxt(
            out_dir / 'ref_data.csv',
            ref_data,
            fmt='%0.3f',
            delimiter=';',
            header=';'.join(ref_labels),
            comments='')

    else:
        pd.DataFrame(index=time_idx, columns=ref_labels, data=ref_data).to_csv(
            out_dir / 'ref_data.csv', sep=';', float_format='%0.3f')

    ref_data = None

    # Simulations
    sim_grp = h5_hdl['data_sim_rltzns']
    for sim_lab in sim_grp:
        sim_data = sim_grp[f'{sim_lab}/data'][...]

        if time_idx is None:
            np.savetxt(
                out_dir / f'sim_data_{sim_lab}.csv',
                sim_data,
                fmt='%0.3f',
                delimiter=';',
                header=';'.join(ref_labels),
                comments='')

        else:
            pd.DataFrame(
                index=time_idx, columns=ref_labels, data=sim_data).to_csv(
                out_dir / f'sim_data_{sim_lab}.csv',
                sep=';',
                float_format='%0.3f')

        sim_data = None

    h5_hdl.close()
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
