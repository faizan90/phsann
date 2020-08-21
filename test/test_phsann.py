'''
@author: Faizan-Uni-Stuttgart
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

    # TODO: Max N plots in plot.py.
    # TODO: Bootstrap plot (densities) for single-site

    main_dir = Path(r'P:\Synchronize\IWS\Testings\fourtrans_practice\phsann')
    os.chdir(main_dir)

    test_unit_peak_flag = False

#==============================================================================
#    Daily HBV sim
#==============================================================================
    in_file_path = r'hbv_sim__1963_2015.csv'

    sim_label = 'test_label_wts_15'  # next:

    labels = 'temp;prec;pet;q_obs'.split(';')

    time_fmt = '%Y-%m-%d'

    beg_time = '1963-01-01'
    end_time = '1963-12-31'

#==============================================================================
#    Daily ppt.
#==============================================================================
#     in_file_path = r'precipitation_bw_1961_2015.csv'
#
#     sim_label = 'test_ms_ppt_03'  # next:
#
# #     labels = ['P1162', 'P1197', 'P4259', 'P5229']
#     labels = ['P1162', 'P5664', 'P1727', 'P5229']
#
#     time_fmt = '%Y-%m-%d'
#
#     beg_time = '2010-01-01'
#     end_time = '2015-12-31'

#==============================================================================
#    Daily
#==============================================================================
#     in_file_path = r'neckar_norm_cop_infill_discharge_1961_2015_20190118.csv'
#
#     sim_label = 'test_lag_opt_27'  # next:
#
#     labels = ['454']  # , '427']  # , '3465']
#
#     time_fmt = '%Y-%m-%d'
#
#     beg_time = '2004-01-01'
#     end_time = '2004-12-31'

#==============================================================================

#==============================================================================
#    Hourly
#==============================================================================
#     in_file_path = r'hourly_bw_discharge__2008__2019.csv'
#
#     sim_label = 'test_mix_dists_33'
#
#     labels = ['3470']  # , '3465']
#
#     time_fmt = '%Y-%m-%d-%H'
#
#     beg_time = '2008-01-01'
#     end_time = '2008-01-31'
#
#==============================================================================

    sep = ';'

    n_vals = 200

    beg_idx = 0
    cen_idx = 50
    end_idx = 199

    verbose = True
#     verbose = False

    h5_name = 'phsann.h5'

    gen_rltzns_flag = True
#     gen_rltzns_flag = False

    plt_flag = True
#     plt_flag = False

    long_test_flag = True
#     long_test_flag = False

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
    asymm_type_1_ms_flag = True
    asymm_type_2_ms_flag = True
    ecop_dens_ms_flag = True
    match_data_ft_flag = True

    scorr_flag = False
    asymm_type_1_flag = False
#     asymm_type_2_flag = False
    ecop_dens_flag = False
    ecop_etpy_flag = False
#     nth_order_diffs_flag = False
    cos_sin_dist_flag = False
    pcorr_flag = False
    asymm_type_1_ms_flag = False
    asymm_type_2_ms_flag = False
    ecop_dens_ms_flag = False
    match_data_ft_flag = False

    n_reals = 4  # A multiple of n_cpus.
    outputs_dir = main_dir / sim_label
    n_cpus = 'auto'

#     lag_steps = np.array([1])
    lag_steps = np.arange(1, 6)
    ecop_bins = 20
    nth_ords = np.arange(1, 5)
    phase_reduction_rate_type = 3
    lag_steps_vld = np.arange(1, 11)
    nth_ords_vld = np.arange(1, 11)

    mag_spec_index_sample_flag = True
#     mag_spec_index_sample_flag = False

    use_dists_in_obj_flag = True
#     use_dists_in_obj_flag = False

    n_beg_phss, n_end_phss = 5, 900
    phs_sample_type = 3
    number_reduction_rate = 0.999
    mult_phs_flag = True
#     mult_phs_flag = False

    wts_flag = True
#     wts_flag = False

#     weights = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0.005], dtype=np.float64)
#     auto_wts_set_flag = False
#     wts_n_iters = None

    weights = None
    auto_wts_set_flag = True
    wts_n_iters = 200

    min_period = None
    max_period = 30

    lags_nths_wts_flag = True
#     lags_nths_wts_flag = False
    lags_nths_exp = 1.5
    lags_nths_n_iters = 200

    label_wts_flag = True
#     label_wts_flag = False
    label_exp = 4
    label_n_iters = 200

    plt_osv_flag = True
    plt_ss_flag = True
    plt_ms_flag = True
    plt_qq_flag = True

#     plt_osv_flag = False
#     plt_ss_flag = False
#     plt_ms_flag = False
#     plt_qq_flag = False

    if long_test_flag:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.999
        update_at_every_iteration_no = 50
        maximum_iterations = int(1e6)
        maximum_without_change_iterations = maximum_iterations
        objective_tolerance = 1e-16
        objective_tolerance_iterations = 5000
        phase_reduction_rate = 0.999
        stop_acpt_rate = 1e-16

        temperature_lower_bound = 1e-2
        temperature_upper_bound = 2000.0
        max_search_attempts = 1000
        n_iterations_per_attempt = update_at_every_iteration_no
        acceptance_lower_bound = 0.6
        acceptance_upper_bound = 0.7
        target_acpt_rate = 0.65
        ramp_rate = 2.0

        acceptance_rate_iterations = 5000
        phase_reduction_rate = 0.999

    else:
        initial_annealing_temperature = 0.0001
        temperature_reduction_ratio = 0.99
        update_at_every_iteration_no = 20
        maximum_iterations = 1
        maximum_without_change_iterations = maximum_iterations
        objective_tolerance = 1e-15
        objective_tolerance_iterations = 20
        phase_reduction_rate = 0.99
        stop_acpt_rate = 1e-15

        temperature_lower_bound = 0.0001
        temperature_upper_bound = 1000.0
        max_search_attempts = 50
        n_iterations_per_attempt = update_at_every_iteration_no
        acceptance_lower_bound = 0.4
        acceptance_upper_bound = 0.5
        target_acpt_rate = 0.45
        ramp_rate = 2.0

        acceptance_rate_iterations = 50
        phase_reduction_rate = 0.95

    if gen_rltzns_flag:
        if test_unit_peak_flag:
            np.random.seed(234324234)

            in_vals_1 = get_unit_peak(
                n_vals, beg_idx, cen_idx + 20, end_idx) + (
                    np.random.random(n_vals) * 0.1)

            in_vals_2 = get_unit_peak(
                n_vals, beg_idx, cen_idx, end_idx) + (
                    np.random.random(n_vals) * 0.1)

#             in_vals_1 = get_unit_peak(
#                 n_vals, beg_idx, cen_idx + 20, end_idx)
#
#             in_vals_2 = get_unit_peak(n_vals, beg_idx, cen_idx, end_idx)

            in_vals = np.concatenate((in_vals_1, in_vals_2)).reshape(-1, 1)

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
            pcorr_flag,
            lag_steps_vld,
            nth_ords_vld,
            asymm_type_1_ms_flag,
            asymm_type_2_ms_flag,
            ecop_dens_ms_flag,
            match_data_ft_flag)

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
            phase_reduction_rate)

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

        if mult_phs_flag:
            phsann_cls.set_mult_phase_settings(
                n_beg_phss, n_end_phss, phs_sample_type, number_reduction_rate)

        if wts_flag:
            phsann_cls.set_objective_weights_settings(
                weights, auto_wts_set_flag, wts_n_iters)

        if np.any([min_period, max_period]):
            phsann_cls.set_selective_phsann_settings(min_period, max_period)

        if lags_nths_wts_flag:
            phsann_cls.set_lags_nths_weights_settings(
                lags_nths_exp, lags_nths_n_iters)

        if label_wts_flag:
            phsann_cls.set_label_weights_settings(
                label_exp, label_n_iters)

        phsann_cls.set_misc_settings(n_reals, outputs_dir, n_cpus)

        phsann_cls.prepare()

        phsann_cls.verify()

        phsann_cls.simulate()

    if plt_flag:
        phsann_plt_cls = PhaseAnnealingPlot(verbose)

        phsann_plt_cls.set_input(
            outputs_dir / h5_name,
            n_cpus,
            plt_osv_flag,
            plt_ss_flag,
            plt_ms_flag,
            plt_qq_flag)

        phsann_plt_cls.set_output(outputs_dir)

        phsann_plt_cls.verify()

        phsann_plt_cls.plot()

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
