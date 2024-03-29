'''
@author: Faizan-Uni-Stuttgart

Nov 29, 2021

2:29:00 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import pandas as pd

DEBUG_FLAG = False


def main():

    main_dir = Path(
        r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann\phd_sims__ppt__quad_phsrand_01')

    os.chdir(main_dir)

    data_dir = Path(r'data_extracted')
    # data_dir = Path(r'resampled_dists__points')

    sep = ';'
    float_fmt = '%0.3f'

    # Can be .pkl or .csv.
    # out_fmt = '.pkl'
    out_fmt = '.csv'

    time_fmt = '%Y-%m-%d'

    # Even number of time steps is output by phsann.
    beg_time = '1999-01-01'
    end_time = '1999-12-30'

    data_time_res = 'D'

    # min_counts correspond to the resolutions. Each resolution when
    # being resampled should have a min-count to get a non Na value.
    # This is because resample sum does not have a skipna flag.
    win_sizes = [2, 7, 14]

#     resample_types = ['mean']  # , 'min', 'max']
    resample_types = ['sum']

    out_dir = Path('resampled_probs__time')
    #==========================================================================

    out_dir.mkdir(exist_ok=True)

    for in_df_path in data_dir.glob('*.csv'):

        print('Going through:', in_df_path.name)

        if in_df_path.suffix == '.csv':
            in_df = pd.read_csv(in_df_path, sep=sep)

        elif in_df_path.suffix == '.pkl':
            in_df = pd.read_pickle(in_df_path)

        else:
            raise NotImplementedError(
                f'Unknown file extension: {in_df_path.suffix}!')

        in_df.index = pd.date_range(beg_time, end_time, freq=data_time_res)

        in_df = in_df.rank(axis=0) / (in_df.shape[0] + 1.0)

        assert isinstance(in_df, pd.DataFrame)
        assert isinstance(in_df.index, pd.DatetimeIndex)

        for win_size in win_sizes:
            for resample_type in resample_types:

                resample_df = getattr(
                    in_df.rolling(win_size, min_periods=win_size, center=True),
                    resample_type)()

                resample_df.dropna(axis=0, how='any', inplace=True)

                out_name = (
                    f'{in_df_path.stem}__'
                    f'WS{win_size}{data_time_res}_RT{resample_type}{out_fmt}')

                out_path = out_dir / out_name

                if out_fmt == '.csv':
                    resample_df.to_csv(
                        out_path,
                        sep=sep,
                        date_format=time_fmt,
                        float_format=float_fmt)

                elif out_fmt == '.pkl':
                    resample_df.to_pickle(out_path)

                else:
                    raise NotImplementedError(
                        f'Unknown file extension: {out_fmt}!')

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
