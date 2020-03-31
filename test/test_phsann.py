'''
@author: Faizan-Uni-Stuttgart

Dec 30, 2019

1:23:33 PM

'''
import os
import sys
import time
import timeit
import traceback as tb
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from phsann import PhaseAnnealing, PhaseAnnealingPlot

# raise Exception

DEBUG_FLAG = False

plt.ioff()


def get_unit_peak(n_vals, beg_index, peak_index, end_index):

    rising_exp = 1.5
    recession_exp = 4

    assert beg_index <= peak_index <= end_index
    assert n_vals > end_index
    assert beg_index >= 0

    unit_peak_arr = np.zeros(n_vals)

    rising_limb = np.linspace(
        0.0, 1.0, peak_index - beg_index, endpoint=False) ** rising_exp

    recessing_limb = np.linspace(
        1.0, 0.0, end_index - peak_index) ** recession_exp

    unit_peak_arr[beg_index:peak_index] = rising_limb
    unit_peak_arr[peak_index:end_index] = recessing_limb

    return unit_peak_arr


def main():

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    test_unit_peak_flag = False

#==============================================================================
#    Daily
#==============================================================================
    in_file_path = r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv'

    sim_label = 'test_sin_cos_dist_01'

    labels = ['420', '454']

    time_fmt = '%Y-%m-%d'

    beg_time = '2005-01-01'
    end_time = '2005-12-31'

    phase_annealing_class_width = 63 * 10000
#==============================================================================

#==============================================================================
#    Hourly
#==============================================================================
#     in_file_path = r'Wannweil_Echaz_2008_2019_hourly_discharge_crctd_fmt.csv'
#
#     sim_label = 'test_dist_in_obj_25_ob0110000_hr'
#
#     labels = ['420']
#
#     time_fmt = '%Y-%m-%d %H:%M:%S'
#
#     beg_time = '2008-01-01'
#     end_time = '2008-12-31'
#
#     phase_annealing_class_width = 200 * 24 * 1
#==============================================================================

    sep = ';'

    n_vals = 200

    beg_idx = 0
    cen_idx = 50
    end_idx = 199

    verbose = True

    h5_name = 'phsann.h5'

    gen_rltzns_flag = True
#     gen_rltzns_flag = False

    plt_flag = True
#     plt_flag = False

    long_test_flag = True
#     long_test_flag = False

    # TODO: generate new phases such that they somehow preserve the properties
    # of the old spectrum. What properties? needs to be investigated.
    # FIXME: The way to minimize difference b/w dists in obj ftns
    # is not distribution fitting but same as getting it exactly right.
    # This might not be what we want. It should be something that shows that
    # they come from the same dist rahter than being the same values.
    # The distance between the sim vals should be uniform when using the
    # function on them.
    # TODO: Investgate, why extrapolate does not work better in obj ftns.
    # TODO: Obj ftns can be computed on a coarser copula that is made finer
    # and finer.
    # TODO: Different time periods' copulas can be compared to get similar
    # features that should be reproduced.
    # TODO: Investigate increase of variance due to extension.

    auto_init_temperature_flag = True
#     auto_init_temperature_flag = False

    scorr_flag = True
    asymm_type_1_flag = True
    asymm_type_2_flag = True
    ecop_dens_flag = True
    ecop_etpy_flag = True
    nth_order_diffs_flag = True
    cos_sin_dist_flag = True
    pcorr_flag = True

    scorr_flag = False
    asymm_type_1_flag = False
    asymm_type_2_flag = False
    ecop_dens_flag = False
    ecop_etpy_flag = False
    nth_order_diffs_flag = False
#     cos_sin_dist_flag = False
    pcorr_flag = False

    n_reals = 5
    outputs_dir = main_dir / sim_label
    n_cpus = 'auto'

    lag_steps = np.array([1, 2, 3, 4, 5])
#     lag_steps = np.arange(1, 121)
    ecop_bins = 50
    nth_ords = np.array([1, 2, 3, 4, 5])
    phase_reduction_rate_type = 3

    mag_spec_index_sample_flag = True
#     mag_spec_index_sample_flag = False

    relative_length = 1
#     relative_length = 2

    use_dists_in_obj_flag = True
    use_dists_in_obj_flag = False

    if long_test_flag:
        initial_annealing_temperature = 0.001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 200
        maximum_iterations = int(2e5)
        maximum_without_change_iterations = 5000
        objective_tolerance = 1e-16
        objective_tolerance_iterations = 1000
        phase_reduction_rate = 0.999
        stop_acpt_rate = 0.0001

        temperature_lower_bound = 1e-5
        temperature_upper_bound = 1000.0
        max_search_attempts = 100
        n_iterations_per_attempt = 3000
        acceptance_lower_bound = 0.5
        acceptance_upper_bound = 0.6
        target_acpt_rate = 0.55
        ramp_rate = 2.0

        acceptance_rate_iterations = 1000
        phase_reduction_rate = 0.999

    else:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 20
        maximum_iterations = 100
        maximum_without_change_iterations = 50
        objective_tolerance = 1e-8
        objective_tolerance_iterations = 20
        phase_reduction_rate = 0.99
        stop_acpt_rate = 0.1

        temperature_lower_bound = 0.0001
        temperature_upper_bound = 1000.0
        max_search_attempts = 50
        n_iterations_per_attempt = 1000  # has to be stable
        acceptance_lower_bound = 0.4
        acceptance_upper_bound = 0.5
        target_acpt_rate = 0.45
        ramp_rate = 2.0

        acceptance_rate_iterations = 50
        phase_reduction_rate = 0.95

    if gen_rltzns_flag:
        if test_unit_peak_flag:
#             in_vals_1 = get_unit_peak(n_vals, beg_idx + 20, cen_idx + 20, end_idx + 10) + (np.random.random(n_vals) * 0.01)
#             in_vals_2 = get_unit_peak(n_vals, beg_idx, cen_idx, end_idx) + (np.random.random(n_vals) * 0.01)

            in_vals_1 = get_unit_peak(
                n_vals, beg_idx, cen_idx + 20, end_idx)

            in_vals_2 = get_unit_peak(n_vals, beg_idx, cen_idx, end_idx)

            in_vals = np.concatenate((in_vals_1, in_vals_2))

        else:
            in_df = pd.read_csv(in_file_path, index_col=0, sep=sep)
            in_df.index = pd.to_datetime(in_df.index, format=time_fmt)

            in_ser = in_df.loc[beg_time:end_time, labels]

            in_vals = in_ser.values

        phsann_cls = PhaseAnnealing(verbose)

        phsann_cls.set_reference_data(in_vals, list(labels))

        phsann_cls.set_objective_settings(
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ecop_etpy_flag,
            nth_order_diffs_flag,
            cos_sin_dist_flag,
            lag_steps,
            ecop_bins,
            nth_ords,
            use_dists_in_obj_flag,
            pcorr_flag)

        phsann_cls.set_annealing_settings(
            initial_annealing_temperature,
            temperature_reduction_ratio,
            update_at_every_iteration_no,
            maximum_iterations,
            maximum_without_change_iterations,
            objective_tolerance,
            objective_tolerance_iterations,
            acceptance_rate_iterations,
            stop_acpt_rate,
            phase_reduction_rate_type,
            mag_spec_index_sample_flag,
            phase_reduction_rate,
            phase_annealing_class_width)

        if auto_init_temperature_flag:
            phsann_cls.set_annealing_auto_temperature_settings(
                temperature_lower_bound,
                temperature_upper_bound,
                max_search_attempts,
                n_iterations_per_attempt,
                acceptance_lower_bound,
                acceptance_upper_bound,
                target_acpt_rate,
                ramp_rate)

        if relative_length != 1:
            phsann_cls.set_extended_length_sim_settings(relative_length)

        phsann_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

        phsann_cls.prepare()

        phsann_cls.verify()

        phsann_cls.simulate()

    if plt_flag:
        phsann_plt_cls = PhaseAnnealingPlot(verbose)

        phsann_plt_cls.set_input(outputs_dir / h5_name, n_cpus)

        phsann_plt_cls.set_output(outputs_dir)

        phsann_plt_cls.verify()

        phsann_plt_cls.plot_opt_state_vars()

        phsann_plt_cls.plot_comparison()

        phsann_plt_cls.plot_validation()

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

    if _save_log_:
        log_link.stop()
