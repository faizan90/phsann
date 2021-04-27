'''
Created on Dec 27, 2019

@author: Faizan
'''
from pathlib import Path

import numpy as np

from ..misc import print_sl, print_el, get_n_cpus

from .data import PhaseAnnealingData as PAD


class PhaseAnnealingSettings(PAD):

    '''Specify settings for the Phase Annealing'''

    def __init__(self, verbose=True):

        PAD.__init__(self, verbose)

        # Order of flags is important. If changed here then synchronize the
        # order wherever necessary.

        # Objective function.
        # For every flag added, increment self._sett_obj_n_flags by one and
        # add the new flag to the _get_all_flags, _set_all_flags_to_one_state
        # and _set_all_flags_to_mult_states of the PhaseAnnealingAlgMisc
        # class. Also, update the _sett_obj_flags_all and
        # _sett_obj_flag_labels array initializations.
        # Modify the _update_wts and related functions accordingly.
        self._sett_obj_scorr_flag = None
        self._sett_obj_asymm_type_1_flag = None
        self._sett_obj_asymm_type_2_flag = None
        self._sett_obj_ecop_dens_flag = None
        self._sett_obj_ecop_etpy_flag = None
        self._sett_obj_nth_ord_diffs_flag = None
        self._sett_obj_cos_sin_dist_flag = None
        self._sett_obj_lag_steps = None
        self._sett_obj_ecop_dens_bins = None
        self._sett_obj_nth_ords = None
        self._sett_obj_use_obj_dist_flag = None
        self._sett_obj_use_obj_lump_flag = None
        self._sett_obj_pcorr_flag = None
        self._sett_obj_lag_steps_vld = None
        self._sett_obj_nth_ords_vld = None
        self._sett_obj_asymm_type_1_ms_flag = None
        self._sett_obj_asymm_type_2_ms_flag = None
        self._sett_obj_ecop_dens_ms_flag = None
        self._sett_obj_match_data_ft_flag = None
        self._sett_obj_match_probs_ft_flag = None
        self._sett_obj_asymm_type_2_ft_flag = None
        self._sett_obj_asymm_type_1_ft_flag = None
        self._sett_obj_nth_ord_diffs_ft_flag = None
        self._sett_obj_asymm_type_1_ms_ft_flag = None
        self._sett_obj_asymm_type_2_ms_ft_flag = None
        self._sett_obj_etpy_ft_flag = None
        self._sett_obj_use_dens_ftn_flag = None
        self._sett_obj_ratio_per_dens_bin = None
        self._sett_obj_etpy_ms_ft_flag = None
        self._sett_obj_n_flags = 22  # 2 additional flags for obj flags.

        self._sett_obj_flag_vals = None
        self._sett_obj_flag_labels = np.array([
            'Spearman correlation (individual)',
            'Asymmetry type 1 (individual)',
            'Asymmetry type 2 (individual)',
            'Empirical copula density (individual)',
            'Empirical copula entropy (individual)',
            'Nth order differences (individual)',
            'Fourier sine-cosine distributions (individual)',
            'Pearson correlation (individual)',
            'Asymmetry type 1 (multisite)',
            'Asymmetry type 2 (multisite)',
            'Empirical copula density (multisite)',
            'Data FT (inidividual)',
            'Probs FT (inidividual)',
            'Asymmetry type 1 FT (individual)',
            'Asymmetry type 2 FT (individual)',
            'Nth order differences FT (individual)',
            'Asymmetry type 1 FT (multisite)',
            'Asymmetry type 2 FT (multisite)',
            'Entropy FT (individual)',
            'Entropy FT (multisite)',
            ])

        # Simulated Annealing.
        self._sett_ann_init_temp = None
        self._sett_ann_temp_red_rate = None
        self._sett_ann_upt_evry_iter = None
        self._sett_ann_max_iters = None
        self._sett_ann_max_iter_wo_chng = None
        self._sett_ann_obj_tol = None
        self._sett_ann_obj_tol_iters = None
        self._sett_ann_acpt_rate_iters = None
        self._sett_ann_stop_acpt_rate = None
        self._sett_ann_phs_red_rate_type = None
        self._sett_ann_phs_red_rate = None
        self._sett_ann_mag_spec_cdf_idxs_flag = None
        self._sett_ann_max_iter_wo_min_updt = None

        # Automatic initialization temperature.
        self._sett_ann_auto_init_temp_temp_bd_lo = None
        self._sett_ann_auto_init_temp_temp_bd_hi = None
        self._sett_ann_auto_init_temp_niters = None
        self._sett_ann_auto_init_temp_acpt_min_bd_lo = 0.2  # Needed for polyfit.
        self._sett_ann_auto_init_temp_acpt_max_bd_hi = 0.8  # Needed for polyfit.
        self._sett_ann_auto_init_temp_acpt_polyfit_n_pts = 5  # Needed for polyfit.
        self._sett_ann_auto_init_temp_acpt_bd_lo = None
        self._sett_ann_auto_init_temp_acpt_bd_hi = None
        self._sett_ann_auto_init_temp_trgt_acpt_rate = None
        self._sett_ann_auto_init_temp_ramp_rate = None
        self._sett_ann_auto_init_temp_n_rltzns = None

        # Multiple phase annealing.
        self._sett_mult_phs_n_beg_phss = None
        self._sett_mult_phs_n_end_phss = None
        self._sett_mult_phs_sample_type = None
        self._sett_mult_phss_red_rate = None

        # Objective function weights.
        self._sett_wts_obj_wts = None
        self._sett_wts_obj_auto_set_flag = None
        self._sett_wts_obj_n_iters = None

        # Selective phase annealing.
        self._sett_sel_phs_min_prd = None
        self._sett_sel_phs_max_prd = None

        # Lags' and nths' weights.
        self._sett_wts_lags_nths_exp = None
        self._sett_wts_lags_nths_n_iters = None
        self._sett_wts_lags_nths_cumm_wts_contrib = None
        self._sett_wts_lags_nths_n_thresh = None
        self._sett_wts_lags_obj_flags = None
        self._sett_wts_nths_obj_flags = None

        # Labels' weights.
        self._sett_wts_label_exp = None
        self._sett_wts_label_n_iters = None

        # CDF penalties.
        self._sett_cdf_pnlt_n_thrsh = None
        self._sett_cdf_pnlt_n_pnlt = None

        # Partial CDF calibration.
        self._sett_prt_cdf_calib_lt = None
        self._sett_prt_cdf_calib_ut = None
        self._sett_prt_cdf_calib_inside_flag = None

        # Misc.
        self._sett_misc_n_rltzns = None
        self._sett_misc_outs_dir = None
        self._sett_misc_n_cpus = None
        self._sett_misc_auto_init_temp_dir = None

        # Flags.
        self._sett_obj_set_flag = False
        self._sett_ann_set_flag = False
        self._sett_auto_temp_set_flag = False
        self._sett_mult_phs_flag = False
        self._sett_wts_obj_set_flag = False
        self._sett_sel_phs_set_flag = False
        self._sett_wts_lags_nths_set_flag = False
        self._sett_wts_label_set_flag = False
        self._sett_cdf_pnlt_set_flag = False
        self._sett_prt_cdf_calib_set_flag = False
        self._sett_misc_set_flag = False
        self._sett_cdf_opt_idxs_flag = False

        self._sett_verify_flag = False
        return

    def set_objective_settings(
            self,
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ecop_etpy_flag,
            nth_order_diffs_flag,
            cos_sin_dist_flag,
            lag_steps,
            ecop_dens_bins,
            nth_ords,
            use_dists_in_obj_flag,
            pcorr_flag,
            lag_steps_vld,
            nth_ords_vld,
            asymm_type_1_ms_flag,
            asymm_type_2_ms_flag,
            ecop_dens_ms_flag,
            match_data_ft_flag,
            match_probs_ft_flag,
            asymm_type_1_ft_flag,
            asymm_type_2_ft_flag,
            nth_order_ft_flag,
            asymm_type_1_ms_ft_flag,
            asymm_type_2_ms_ft_flag,
            etpy_ft_flag,
            use_dens_ftn_flag,
            ratio_per_dens_bin,
            etpy_ms_ft_flag):

        '''
        Type of objective functions to use and their respective inputs.
        NOTE: all non-flag parameters have to be specified even if their
        flags are False.

        Parameters
        ----------
        scorr_flag : bool
            Whether to minimize the differences between the spearman
            correlations of the reference and generated realizations.
        asymm_type_1_flag : bool
            Whether to minimize the differences between the first type
            normalized asymmetry of the reference and generated realizations.
        asymm_type_2_flag : bool
            Whether to minimize the differences between the second type
            normalized asymmetry of the reference and generated realizations.
        ecop_dens_flag : bool
            Whether to minimize the differences between the reference and
            simulated empirical copula densities.
        ecop_etpy_flag : bool
            Whether to minimize the differences between reference and
            simulated empirical copula density entropies.
        nth_order_diffs_flag : bool
            Whether to minimize nth order differences distribution at given
            nth_ords between reference and simulated series.
        cos_sin_dist_flag : bool
            Whether to match the real and imaginary parts' distribution
            of the reference and simulated.
        lag_steps : 1D integer np.ndarray
            The lagged steps at which to evaluate the objective functions.
            All should be greater than zero and unique. Maximum should be
            < 1000.
        ecop_dens_bins : integer
            Number of bins for the empirical copula. The more the bins the
            finer the density match between the reference and simulated.
        nth_ords : 1D integer np.ndarray
            Order(s) of differences (1st, 2nd, ...) if nth_order_diffs_flag
            is True. Maximum should be < 1000.
        use_dists_in_obj_flag : bool
            Whether to minimize the difference of objective function values
            or the distributions that make that value up. e.g. For asymmetry,
            if the flag is False, then the value of asymmetry if matched else,
            the distribution of the value that produced the asymmetry is
            brought close to the reference.
        pcorr_flag : bool
            Whether to minimize the differences between the pearson
            correlations of the reference and generated realizations. This is
            irrelevant for phase annealing only. In case of magnitude
            annealing, pearson correlation is lost and must be optimized for.
        lag_steps_vld : 1D integer np.ndarray
            Same as lag steps but these steps are used for plots. Computed
            at the end of simulation. Also, it is unionized with lag_steps.
             Maximum should be < 1000.
        nth_ords_vld : 1D integer np.ndarray
            Same as nth_ords but these orders are used for plots. Computed
            at the end of simulation. Also, it is unionized with nth_ords.
             Maximum should be < 1000.
        asymm_type_1_ms_flag : bool
            Whether to minimize the differences between the first type
            normalized asymmetry of the reference and generated realizations,
            for multisite copulas.
        asymm_type_2_ms_flag : bool
            Whether to minimize the differences between the second type
            normalized asymmetry of the reference and generated realizations,
            for multisite copulas.
        ecop_dens_ms_flag : bool
            Whether to minimize the differences between the reference and
            simulated empirical copula densities, for multisite copulas.
        match_data_ft_flag : bool
            Whether to match the amplitude spectrum of the simulated series'
            data with that of the reference. For certain objective functions,
            it can happen that the amplitude spectrum of the simulations is
            very different than that of the reference.
        match_probs_ft_flag : bool
            Whether to match the amplitude spectrum of the simulated series'
            probabilities with that of the reference. For certain objective
            functions, it can happen that the amplitude spectrum of the
            simulations is very different than that of the reference.
        asymm_type_1_ft_flag : bool
            Whether to match the amplitude spectrum of the simulated series'
            asymmetry 1 numerator series with that of the reference.
        asymm_type_2_ft_flag : bool
            Whether to match the amplitude spectrum of the simulated series'
            asymmetry 2 numerator series with that of the reference.
        nth_order_ft_flag : bool
            Whether to match the amplitude spectrum of the simulated series'
            nth order series with that of the reference.
        asymm_type_1_ms_ft_flag : bool
            Multisite version of asymm_type_1_ft_flag.
        asymm_type_2_ms_ft_flag : bool
            Multisite version of asymm_type_2_ft_flag.
        etpy_ft_flag : bool
            Whether to match the power spectrum of the simulated active
            information entropy series to that of the reference.
        use_dens_ftn_flag : bool
            Whether to use empirical density function matching for various
            objective functions when use_dists_in_obj_flag is True. This
            becomes important for objective function like asymm2, where
            very small values near zero dominate the distribution and have
            no effect on the variance. use_dists_in_obj_flag should be True.
        ratio_per_dens_bin : float
            Relative values to use per bin of the empirical density function.
            Should be greater than zero and less than one.
            use_dists_in_obj_flag and use_dens_ftn_flag should be True.
        etpy_ms_ft_flag : bool
            Multisite version of etpy_ft_flag.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting objective function settings for phase annealing...\n')

        assert isinstance(scorr_flag, bool), 'scorr_flag not a boolean!'

        assert isinstance(asymm_type_1_flag, bool), (
            'asymm_type_1_flag not a boolean!')

        assert isinstance(asymm_type_2_flag, bool), (
            'asymm_type_2_flag not a boolean!')

        assert isinstance(ecop_dens_flag, bool), (
            'ecop_dens_flag not a boolean!')

        assert isinstance(ecop_etpy_flag, bool), (
            'ecop_etpy_flag not a boolean!')

        assert isinstance(nth_order_diffs_flag, bool), (
            'nth_order_diffs_flag not a boolean!')

        assert isinstance(cos_sin_dist_flag, bool), (
            'cos_sin_dist_flag not a boolean!')

        assert isinstance(pcorr_flag, bool), (
            'pcorr_flag not a boolean!')

        assert isinstance(asymm_type_1_ms_flag, bool), (
            'asymm_type_1_ms_flag not a boolean!')

        assert isinstance(asymm_type_2_ms_flag, bool), (
            'asymm_type_2_ms_flag not a boolean!')

        assert isinstance(ecop_dens_ms_flag, bool), (
            'ecop_dens_ms_flag not a boolean!')

        assert isinstance(match_data_ft_flag, bool), (
            'match_data_ft_flag not a boolean!')

        assert isinstance(match_probs_ft_flag, bool), (
            'match_probs_ft_flag not a boolean!')

        assert isinstance(asymm_type_1_ft_flag, bool), (
            'asymm_type_1_ft_flag not a boolean!')

        assert isinstance(asymm_type_2_ft_flag, bool), (
            'asymm_type_2_ft_flag not a boolean!')

        assert isinstance(nth_order_ft_flag, bool), (
            'nth_order_ft_flag not a boolean!')

        assert isinstance(asymm_type_1_ms_ft_flag, bool), (
            'asymm_type_1_ms_ft_flag not a boolean!')

        assert isinstance(asymm_type_2_ms_ft_flag, bool), (
            'asymm_type_2_ms_ft_flag not a boolean!')

        assert isinstance(etpy_ft_flag, bool), (
            'etpy_ft_flag not a boolean!')

        assert isinstance(use_dens_ftn_flag, bool), (
            'use_dens_ftn_flag not a boolean!')

        assert isinstance(ratio_per_dens_bin, float), (
            'ratio_per_dens_bin not a float!')

        assert isinstance(etpy_ms_ft_flag, bool), (
            'etpy_ms_ft_flag not a boolean!')

        assert any([
            scorr_flag,
            asymm_type_1_flag,
            asymm_type_2_flag,
            ecop_dens_flag,
            ecop_etpy_flag,
            nth_order_diffs_flag,
            cos_sin_dist_flag,
            pcorr_flag,
            asymm_type_1_ms_flag,
            asymm_type_2_ms_flag,
            ecop_dens_ms_flag,
            match_data_ft_flag,
            match_probs_ft_flag,
            asymm_type_1_ft_flag,
            asymm_type_2_ft_flag,
            nth_order_ft_flag,
            asymm_type_1_ms_ft_flag,
            asymm_type_2_ms_ft_flag,
            etpy_ft_flag,
            etpy_ms_ft_flag,
            ]), 'All objective function flags are False!'

        assert isinstance(lag_steps, np.ndarray), (
            'lag_steps not a numpy arrray!')

        assert lag_steps.ndim == 1, 'lag_steps not a 1D array!'
        assert lag_steps.size > 0, 'lag_steps is empty!'
        assert lag_steps.dtype == np.int, 'lag_steps has a non-integer dtype!'
        assert np.all(lag_steps > 0), 'lag_steps has non-postive values!'

        assert np.all(lag_steps < 1000), 'Invalid lag_step value(s)!'

        assert np.unique(lag_steps).size == lag_steps.size, (
            'Non-unique values in lag_steps!')

        assert isinstance(ecop_dens_bins, int), (
            'ecop_dens_bins is not an integer!')

        assert ecop_dens_bins > 1, 'Invalid ecop_dens_bins!'

        assert isinstance(nth_ords, np.ndarray), (
            'nth_ords not a numpy ndarray!')

        assert nth_ords.ndim == 1, 'nth_ords not a 1D array!'
        assert nth_ords.size > 0, 'nth_ords is empty!'
        assert np.all(nth_ords > 0), 'nth_ords has non-postive values!'
        assert nth_ords.dtype == np.int, 'nth_ords has a non-integer dtype!'

        assert np.all(nth_ords < 1000), 'Invalid nth_ord value(s)!'

        assert np.unique(nth_ords).size == nth_ords.size, (
            'Non-unique values in nth_ords!')

        assert nth_ords.max() < 1000, (
            'Maximum of nth_ords cannot be more than 1000!')

        assert isinstance(use_dists_in_obj_flag, bool), (
            'use_dists_in_obj_flag not a boolean!')

        assert isinstance(lag_steps_vld, np.ndarray), (
            'lag_steps_vld not a numpy arrray!')

        assert lag_steps_vld.ndim == 1, 'lag_steps_vld not a 1D array!'
        assert lag_steps_vld.size > 0, 'lag_steps_vld is empty!'
        assert lag_steps_vld.dtype == np.int, (
            'lag_steps_vld has a non-integer dtype!')

        assert np.all(lag_steps_vld < 1000), 'Invalid lag_step_vld value(s)!'

        assert np.all(lag_steps_vld > 0), (
            'lag_steps_vld has non-postive values!')

        assert np.unique(lag_steps_vld).size == lag_steps_vld.size, (
            'Non-unique values in lag_steps_vld!')

        assert isinstance(nth_ords_vld, np.ndarray), (
            'nth_ords_vld not a numpy ndarray!')

        assert nth_ords_vld.ndim == 1, 'nth_ords_vld not a 1D array!'
        assert nth_ords_vld.size > 0, 'nth_ords_vld is empty!'
        assert np.all(nth_ords_vld > 0), 'nth_ords_vld has non-postive values!'
        assert nth_ords_vld.dtype == np.int, (
            'nth_ords_vld has a non-integer dtype!')

        assert np.all(nth_ords_vld < 1000), 'Invalid nth_ord_vld value(s)!'

        assert np.unique(nth_ords_vld).size == nth_ords_vld.size, (
            'Non-unique values in nth_ords_vld!')

        assert nth_ords_vld.max() < 1000, (
            'Maximum of nth_ords_vld cannot be more than 1000!')

        if use_dens_ftn_flag:
            assert use_dists_in_obj_flag, (
                'use_dists_in_obj_flag must be set when '
                'use_dens_ftn_flag is set!')

        assert 0 < ratio_per_dens_bin < 1, (
            'Invalid ratio_per_dens_bin!')

        self._sett_obj_scorr_flag = scorr_flag
        self._sett_obj_asymm_type_1_flag = asymm_type_1_flag
        self._sett_obj_asymm_type_2_flag = asymm_type_2_flag
        self._sett_obj_ecop_dens_flag = ecop_dens_flag
        self._sett_obj_ecop_etpy_flag = ecop_etpy_flag
        self._sett_obj_nth_ord_diffs_flag = nth_order_diffs_flag
        self._sett_obj_cos_sin_dist_flag = cos_sin_dist_flag
        self._sett_obj_lag_steps = np.sort(lag_steps).astype(np.int64)
        self._sett_obj_ecop_dens_bins = ecop_dens_bins
        self._sett_obj_nth_ords = np.sort(nth_ords).astype(np.int64)
        self._sett_obj_use_obj_dist_flag = use_dists_in_obj_flag
        self._sett_obj_use_obj_lump_flag = not self._sett_obj_use_obj_dist_flag
        self._sett_obj_pcorr_flag = pcorr_flag
        self._sett_obj_asymm_type_1_ms_flag = asymm_type_1_ms_flag
        self._sett_obj_asymm_type_2_ms_flag = asymm_type_2_ms_flag
        self._sett_obj_ecop_dens_ms_flag = ecop_dens_ms_flag
        self._sett_obj_match_data_ft_flag = match_data_ft_flag
        self._sett_obj_match_probs_ft_flag = match_probs_ft_flag
        self._sett_obj_asymm_type_1_ft_flag = asymm_type_1_ft_flag
        self._sett_obj_asymm_type_2_ft_flag = asymm_type_2_ft_flag
        self._sett_obj_nth_ord_diffs_ft_flag = nth_order_ft_flag
        self._sett_obj_asymm_type_1_ms_ft_flag = asymm_type_1_ms_ft_flag
        self._sett_obj_asymm_type_2_ms_ft_flag = asymm_type_2_ms_ft_flag
        self._sett_obj_etpy_ft_flag = etpy_ft_flag
        self._sett_obj_use_dens_ftn_flag = use_dens_ftn_flag
        self._sett_obj_ratio_per_dens_bin = ratio_per_dens_bin
        self._sett_obj_etpy_ms_ft_flag = etpy_ms_ft_flag

        self._sett_obj_lag_steps_vld = np.sort(np.union1d(
            self._sett_obj_lag_steps, lag_steps_vld.astype(np.int64)))

        self._sett_obj_nth_ords_vld = np.sort(np.union1d(
            self._sett_obj_nth_ords, nth_ords_vld.astype(np.int64)))

        if self._vb:
            print(
                'Rank correlation flag:',
                self._sett_obj_scorr_flag)

            print(
                'Asymmetry type 1 flag:',
                self._sett_obj_asymm_type_1_flag)

            print(
                'Asymmetry type 2 flag:',
                self._sett_obj_asymm_type_2_flag)

            print(
                'Empirical copula density flag:',
                self._sett_obj_ecop_dens_flag)

            print(
                'Empirical copula entropy flag:',
                self._sett_obj_ecop_etpy_flag)

            print(
                'Nth order differences flag:',
                self._sett_obj_nth_ord_diffs_flag)

            print(
                'Cosine and Sine distribution flag:',
                self._sett_obj_cos_sin_dist_flag)

            print(
                'Lag steps:',
                self._sett_obj_lag_steps)

            print(
                'Empirical copula density bins:',
                self._sett_obj_ecop_dens_bins)

            print(
                'Nth orders:',
                self._sett_obj_nth_ords)

            print(
                'Fit distributions in objective functions flag:',
                self._sett_obj_use_obj_dist_flag)

            print(
                'Pearson correrlation flag:',
                self._sett_obj_pcorr_flag)

            print(
                'Validation lag steps:',
                self._sett_obj_lag_steps_vld)

            print(
                'Validation Nth orders:',
                self._sett_obj_nth_ords_vld)

            print(
                'Multisite asymmetry 1 flag:',
                self._sett_obj_asymm_type_1_ms_flag)

            print(
                'Multisite asymmetry 2 flag:',
                self._sett_obj_asymm_type_2_ms_flag)

            print(
                'Multisite empirical copula density flag:',
                self._sett_obj_ecop_dens_ms_flag)

            print(
                'Match data FT flag:',
                self._sett_obj_match_data_ft_flag)

            print(
                'Match probs FT flag:',
                self._sett_obj_match_probs_ft_flag)

            print(
                'Asymmetry type 1 FT flag:',
                self._sett_obj_asymm_type_1_ft_flag)

            print(
                'Asymmetry type 2 FT flag:',
                self._sett_obj_asymm_type_2_ft_flag)

            print(
                'Nth order differences FT flag:',
                self._sett_obj_nth_ord_diffs_ft_flag)

            print(
                'Multisite asymmetry 1 FT flag:',
                self._sett_obj_asymm_type_1_ms_ft_flag)

            print(
                'Multisite asymmetry 2 FT flag:',
                self._sett_obj_asymm_type_2_ms_ft_flag)

            print(
                'Match entropy FT flag:',
                self._sett_obj_etpy_ft_flag)

            print(
                'Use density function in objective functions:',
                self._sett_obj_use_dens_ftn_flag)

            print(
                'Ratio per density function bin:',
                self._sett_obj_ratio_per_dens_bin)

            print(
                'Multisite entropy FT flag:',
                self._sett_obj_etpy_ms_ft_flag)

            print_el()

        self._sett_obj_set_flag = True
        return

    def set_annealing_settings(
            self,
            initial_annealing_temperature,
            temperature_reduction_rate,
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
            maximum_iterations_without_updating_best):

        '''
        Simulated annealing algorithm parameters

        Parameters
        ----------
        initial_annealing_temperature : float
            The starting temperature of the annealing temperature. Should be
            > 0 and < infinity!
        temperature_reduction_rate : float
            The rate by which to reduce the temperature after
            update_at_every_iteration_no have passed. Should be > 0 and <= 1.
        update_at_every_iteration_no : integer
            When to update the temperature. Should be > 0.
        maximum_iterations : integer
            Number of iterations at maximum if no other stopping criteria is
            met. Should be > 0.
        maximum_without_change_iterations : integer
            To stop looking for an optimum after
            maximum_without_change_iterations consecutive iterations do not
            yield a better optimum. Should be > 0.
        objective_tolerance : float
            To stop the optimization if mean of the absolute differences
            between consecutive objective_tolerance_iterations iterations
            is less than or equal to objective_tolerance. Should be >= 0.
        objective_tolerance_iterations : integer
            See the parameter objective_tolerance. Should be > 0.
        acceptance_rate_iterations : integer
            Number of iterations to take for mean acceptance rate. Should be
            greater than 0.
        stop_acpt_rate : float
            The acceptance rate at or below which the optimization stops.
            Should be >= 0 and <= 1.
        phase_reduction_rate_type : integer
            How to limit the magnitude of the newly generated phases.
            A number between 0 and 3.
            0:    No limiting performed.
            1:    A linear reduction with respect to the maximum iterations is
                  applied. The more the iteration number the less the change.
            2:    Reduce the rate by multiplying the previous rate by
                  phase_reduction_rate.
            3:    Reduction rate is equal to the mean acceptance rate of
                  previous acceptance_rate_iterations.
        mag_spec_index_sample_flag : bool
            Whether to sample new freqeuncy indices on a magnitude spectrum
            CDF based weighting i.e. frequencies having more amplitude
            have a bigger chance of being sampled.
        phase_reduction_rate : float
            If phase_reduction_rate_type is 2, then the new phase reduction
            rate is previous multiplied by phase_reduction_rate_type. Should
            be > 0 and <= 1.
        maximum_iterations_without_updating_best : int
            Maximum number of iterations without updating the global best
            solution. This is important for cases where the optimization
            stagnates, keeps on updating the current solution but never
            the global minimum. Should be >= 0.
        '''

        if self._vb:
            print_sl()

            print('Setting annealing settings for phase annealing...\n')

        assert isinstance(initial_annealing_temperature, float), (
            'initial_annealing_temperature not a float!')

        assert isinstance(temperature_reduction_rate, float), (
            'temperature_reduction_rate not a float!')

        assert isinstance(update_at_every_iteration_no, int), (
            'update_at_every_iteration_no not an integer!')

        assert isinstance(maximum_iterations, int), (
            'maximum_iterations not an integer!')

        assert isinstance(maximum_without_change_iterations, int), (
            'maximum_without_change_iterations not an integer!')

        assert isinstance(objective_tolerance, float), (
            'objective_tolerance not a float!')

        assert isinstance(objective_tolerance_iterations, int), (
            'objective_tolerance_iterations not an integer!')

        assert isinstance(acceptance_rate_iterations, int), (
            'acceptance_rate_iterations not an integer!')

        assert isinstance(stop_acpt_rate, float), (
            'stop_acpt_rate not a float!')

        assert isinstance(phase_reduction_rate_type, int), (
            'phase_reduction_rate_type not an integer!')

        assert 0 < initial_annealing_temperature < np.inf, (
            'Invalid initial_annealing_temperature!')

        assert 0 < temperature_reduction_rate <= 1, (
            'Invalid temperature_reduction_rate!')

        assert maximum_iterations >= 0, 'Invalid maximum_iterations!'

        assert update_at_every_iteration_no >= 0, (
            'Invalid update_at_every_iteration_no!')

        assert maximum_without_change_iterations >= 0, (
            'Invalid maximum_without_change_iterations!')

        assert 0 <= objective_tolerance <= np.inf, (
            'Invalid objective_tolerance!')

        assert objective_tolerance_iterations >= 0, (
            'Invalid objective_tolerance_iterations!')

        assert acceptance_rate_iterations >= 0, (
            'Invalid acceptance_rate_iterations!')

        assert 0 <= stop_acpt_rate <= 1.0, (
            'Invalid stop_acpt_rate!')

        assert 0 <= phase_reduction_rate_type <= 3, (
            'Invalid phase_reduction_rate_type!')

        assert isinstance(mag_spec_index_sample_flag, bool), (
            'mag_spec_index_sample_flag not a boolean!')

        if phase_reduction_rate_type == 2:

            assert isinstance(phase_reduction_rate, float), (
                'phase_reduction_rate is not a float!')

            assert 0 < phase_reduction_rate <= 1, (
                'Invalid phase_reduction_rate!')

        elif phase_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError('Unknown phase_reduction_rate_type!')

        assert isinstance(
            maximum_iterations_without_updating_best, int), (
                'maximum_iterations_without_updating_best not an integer!')

        assert maximum_iterations_without_updating_best >= 0, (
            'Invalid maximum_iterations_without_updating_best!')

        self._sett_ann_init_temp = initial_annealing_temperature
        self._sett_ann_temp_red_rate = temperature_reduction_rate
        self._sett_ann_upt_evry_iter = update_at_every_iteration_no
        self._sett_ann_max_iters = maximum_iterations
        self._sett_ann_max_iter_wo_chng = maximum_without_change_iterations
        self._sett_ann_obj_tol = objective_tolerance
        self._sett_ann_obj_tol_iters = objective_tolerance_iterations
        self._sett_ann_acpt_rate_iters = acceptance_rate_iterations
        self._sett_ann_stop_acpt_rate = stop_acpt_rate
        self._sett_ann_phs_red_rate_type = phase_reduction_rate_type
        self._sett_ann_mag_spec_cdf_idxs_flag = mag_spec_index_sample_flag

        if phase_reduction_rate_type == 2:
            self._sett_ann_phs_red_rate = phase_reduction_rate

        elif phase_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError('Unknown phase_reduction_rate_type!')

        self._sett_ann_max_iter_wo_min_updt = (
            maximum_iterations_without_updating_best)

        if self._vb:

            print(
                'Initial annealing temperature:', self._sett_ann_init_temp)

            print(
                'Temperature reduction rate:', self._sett_ann_temp_red_rate)

            print(
                'Temperature update iteration:', self._sett_ann_upt_evry_iter)

            print(
                'Maximum iterations:', self._sett_ann_max_iters)

            print(
                'Maximum iterations without change:',
                self._sett_ann_max_iter_wo_chng)

            print(
                'Objective function tolerance:',
                self._sett_ann_obj_tol)

            print(
                'Objective function tolerance iterations:',
                self._sett_ann_obj_tol_iters)

            print(
                'Acceptance rate iterations:',
                self._sett_ann_acpt_rate_iters)

            print(
                'Stopping acceptance rate:',
                self._sett_ann_stop_acpt_rate)

            print(
                'Phase reduction rate type:',
                self._sett_ann_phs_red_rate_type)

            print(
                'Magnitude spectrum based indexing flag:',
                self._sett_ann_mag_spec_cdf_idxs_flag)

            print(
                'Phase reduction rate:', self._sett_ann_phs_red_rate)

            print(
                'Maximum iteration without updating the global minimum:',
                self._sett_ann_max_iter_wo_min_updt)

            print_el()

        self._sett_ann_set_flag = True
        return

    def set_annealing_auto_temperature_settings(
            self,
            temperature_lower_bound,
            temperature_upper_bound,
            n_iterations_per_attempt,
            acceptance_lower_bound,
            acceptance_upper_bound,
            target_acceptance_rate,
            ramp_rate):

        '''
        Automatic annealing initial temperature search parameters. All
        realizations get the same initial temperature. It is computed by
        ramping up from the temperature_lower_bound to temperature_upper_bound.
        After several evaluations, a line is fitted to acceptance rates and
        temperatures, the initialization temperature is then interpolated
        at the target acceptance rate. An AssertionError is raised if
        no aceptance rate is such that it is inbetween acceptance_lower_bound
        and acceptance_upper_bound.

        Plots of the initialzation temperatures versus their corresponding
        acceptance rates are saved in the directory
        "auto_init_temps__acpt_rates" after their computation i.e. before the
        actual optimization takes place. This is done so that the user can
        take a look at the initialization situation before the optimization
        goes forth.

        A self._sett_ann_auto_init_temp_acpt_polyfit_n_pts number of valid
        points is required for the line fitting to take place. An exception
        is raised if required points are not there. In that case, change
        the search settings such that more points are found in between
        self._sett_ann_auto_init_temp_acpt_min_bd_lo and
        self._sett_ann_auto_init_temp_acpt_max_bd_hi.

        Parameters
        ----------
        temperature_lower_bound : float
            Lower bound of the temperature search space. Should be > 0
        temperature_upper_bound : float
            Upper bound of the temperature search space. Should be >
            temperature_lower_bound and < infinity.
        n_iterations_per_attempt : integer
            Number of times to run the annealing algorithm after which to
            compute the mean acceptance rate. Should be large enough to
            give a stable acceptance rate. Should be > 0.
        acceptance_lower_bound : float
            Lower bounds of the acceptance rate for an initial starting
            temperature to be accepted for the optimization.
            Should be >= self._sett_ann_auto_init_temp_acpt_min_bd_lo and <
            acceptance_upper_bound.
            During the search any acceptance rate below
            self._sett_ann_auto_init_temp_acpt_min_bd_lo is not taken.
        acceptance_upper_bound : float
            Upper bounds of the acceptance rate for an initial starting
            temperature to be accepted for the optimization. Should be >
            acceptance_lower_bound and <=
            self._sett_ann_auto_init_temp_acpt_max_bd_hi. During the
            search any acceptance rate above
            self._sett_ann_auto_init_temp_acpt_max_bd_hi is not taken.
        target_acceptance_rate : float
            The optimum acceptance rate for which to find the initial
            temperature. Has to be in between acceptance_lower_bound and
            acceptance_upper_bound. The final temperature is selected based
            on minimum distance from the target_acceptance_rate.
        ramp_rate : float
            The rate at which to increase/ramp the temperature every
            n_iterations_per_attempt. Temperature is ramped up from
            acceptance_lower_bound to acceptance_upper_bound. Next iteration's
            temperature = previous * ramp_rate. The search stops if
            n_iterations_per_attempt are reached or next temperature
            is greater than temperature_upper_bound or acceptance rate is 1.
            Should be > 1 and < infinity.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting automatic annealing initial temperature settings '
                'for phase annealing...\n')

        assert isinstance(temperature_lower_bound, float), (
            'temperature_lower_bound not a float!')

        assert isinstance(temperature_upper_bound, float), (
            'temperature_upper_bound not a float!')

        assert isinstance(n_iterations_per_attempt, int), (
            'n_iterations_per_attempt not an integer!')

        assert isinstance(acceptance_lower_bound, float), (
            'acceptance_lower_bound not a float!')

        assert isinstance(acceptance_upper_bound, float), (
            'acceptance_upper_bound not a float!')

        assert isinstance(target_acceptance_rate, float), (
            'target_acceptance_rate not a float!')

        assert isinstance(ramp_rate, float), 'ramp_rate not a float!'

        assert (
            0 < temperature_lower_bound < temperature_upper_bound < np.inf), (
                'Inconsistent or invalid temperature_lower_bound and '
                'temperature_upper_bound!')

        assert 0 < n_iterations_per_attempt, (
            'Invalid n_iterations_per_attempt!')

        assert (
            self._sett_ann_auto_init_temp_acpt_min_bd_lo <
            acceptance_lower_bound <=
            target_acceptance_rate <=
            acceptance_upper_bound <=
            self._sett_ann_auto_init_temp_acpt_max_bd_hi), (
                'Invalid or inconsistent acceptance_lower_bound, '
                'target_acceptance_rate or acceptance_upper_bound!')

        assert 1 < ramp_rate < np.inf, 'Invalid ramp_rate!'

        self._sett_ann_auto_init_temp_temp_bd_lo = temperature_lower_bound
        self._sett_ann_auto_init_temp_temp_bd_hi = temperature_upper_bound
        self._sett_ann_auto_init_temp_niters = n_iterations_per_attempt
        self._sett_ann_auto_init_temp_acpt_bd_lo = acceptance_lower_bound
        self._sett_ann_auto_init_temp_acpt_bd_hi = acceptance_upper_bound
        self._sett_ann_auto_init_temp_trgt_acpt_rate = target_acceptance_rate
        self._sett_ann_auto_init_temp_ramp_rate = ramp_rate

        if self._vb:
            print(
                'Lower tmeperature bounds:',
                self._sett_ann_auto_init_temp_temp_bd_lo)

            print(
                'Upper temperature bounds:',
                self._sett_ann_auto_init_temp_temp_bd_hi)

            print(
                'Number of iterations per attempt:',
                self._sett_ann_auto_init_temp_niters)

            print(
                'Lower acceptance bounds:',
                self._sett_ann_auto_init_temp_acpt_bd_lo)

            print(
                'Upper acceptance bounds:',
                self._sett_ann_auto_init_temp_acpt_bd_hi)

            print(
                'Temperature ramp rate:',
                self._sett_ann_auto_init_temp_ramp_rate
                )

            print(
                'Target acceptance rate:',
                self._sett_ann_auto_init_temp_trgt_acpt_rate)

            print_el()

        self._sett_auto_temp_set_flag = True
        return

    def set_mult_phase_settings(
            self,
            n_beg_phss,
            n_end_phss,
            sample_type,
            number_reduction_rate):

        '''
        Randomize multiple phases instead of just one.

        A random number of phases are generated for each iteration between
        n_beg_phss and n_end_phss (both inclusive). These values are adjusted
        if available phases/magnitudes are not enough, internally but these
        values are kept.

        Parameters
        ----------
        n_beg_phss : integer
            Minimum phases/magnitudes to randomize per iteration.
            Should be > 0.
        n_end_phss : integer
            Maximum number of phases/magnitudes to randomize per iteration.
            Should be >= n_beg_phss.
        sample_type : integer
            How to sample the number of phases generated for each iteration.
            0:  New phase indices are generated randomly between
                n_beg_phss and n_end_phss, regardless of where in the
                optimization for each iteration.
            1:  The number of newly generated phases depends on the ratio
                of current iteration number and maximum_iterations.
            2:  The number of newly generated phase indices is reduced
                by multiplying with number_reduction_rate at every
                temperature update iteration.
            3:  The number of newly generated phase indices is proportional
                to the acceptance rate.
        number_reduction_rate : float
            Generated phase indices reduction rate. A value between > 0 and
            <= 1. The same as temperature reduction schedule. Required
            to have a valid value only is sample_type == 2.

        NOTE: In case the difference between n_beg_phss and n_end_phss
        is high and mag_spec_index_sample_flag is True and the
        distribution of the magnitude spectrum is highly skewed, it
        will take a while to get the indices (per iteration). So it
        might be a good idea to set mag_spec_index_sample_flag to False.
        '''

        if self._vb:
            print_sl()

            print('Settings multiple phase annealing settings...\n')

        assert isinstance(n_beg_phss, int), 'n_beg_phss not an integer!'
        assert isinstance(n_end_phss, int), 'n_end_phss not an integer!'
        assert isinstance(sample_type, int), 'sample_type is not an integer!'

        assert n_beg_phss > 0, 'Invalid n_beg_phss!'
        assert n_end_phss >= n_beg_phss, 'Invalid n_end_phss!'

        assert sample_type in (0, 1, 2, 3), 'Invalid sample_type!'

        if sample_type > 0:
            assert n_beg_phss < n_end_phss, (
                'n_beg_phss and n_end_phss cannot be equal for sample_type '
                '> 0!')

        if sample_type == 2:
            assert isinstance(number_reduction_rate, float), (
                'number_reduction_rate not a float!')
            assert 0 < number_reduction_rate <= 1, (
                'Invalid number_reduction_rate!')

        elif sample_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError

        self._sett_mult_phs_n_beg_phss = n_beg_phss
        self._sett_mult_phs_n_end_phss = n_end_phss
        self._sett_mult_phs_sample_type = sample_type

        if sample_type == 2:
            self._sett_mult_phss_red_rate = number_reduction_rate

        if self._vb:
            print(
                f'Starting multiple phase indices: '
                f'{self._sett_mult_phs_n_beg_phss}')

            print(
                f'Ending multiple phase indices: '
                f'{self._sett_mult_phs_n_end_phss}')

            print(
                f'Multiple phase sampling type: '
                f'{self._sett_mult_phs_sample_type}')

            print(
                f'Multiple phase number reduction rate: '
                f'{self._sett_mult_phss_red_rate}')

            print_el()

        self._sett_mult_phs_flag = True
        return

    def set_objective_weights_settings(
            self,
            weights,
            auto_wts_set_flag,
            wts_n_iters):

        f'''
        Set weights for all objective functions regardless of their use.

        Weights are needed because different objective functions
        have different magnitudes. Without weights, the optimization sees only
        the objective function with the highest magnitude thereby ignoring
        smaller magnitude objective functions.

        Two possibilites are provided. Based on the state of the
        auto_wts_set_flag i.e. Manual specification and automatic weights
        detection.

        Parameters
        ----------
        weights : 1D float64 np.ndarray or None
            Manual specification of weights for each objective function.
            Should be equal to the number of objective functions available i.e.
            {self._sett_obj_flag_labels.size}. The order corresponds to that
            of the flags in set_objective_settings. If weights is not None,
            then auto_wts_set_flag must be False, and wts_n_iters must be
            None.
        auto_wts_set_flag : bool
            Whether to detect weights automatically. If automatic initial
            temperature detection is set, then weights are computed before
            the annealing starts by calling the objective function with full
            phase spectrum randomization wts_n_iters times, mean of the
            values are taken afterwards to get the weight for each objective
            function.
            If True then weights must be None and wts_n_iters must be
            specified.
        wts_n_iters : int
            The number of times to call the objective function to get an
            estimate of the weights that each function should have.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting objective function weights settings '
                'for phase annealing...\n')

        if weights is not None:
            assert isinstance(weights, np.ndarray)
            assert weights.ndim == 1
            assert weights.size == self._sett_obj_flag_labels.size
            assert np.all(np.isfinite(weights))
            assert weights.dtype == np.float64

            assert auto_wts_set_flag is False

            assert wts_n_iters is None

        else:
            assert auto_wts_set_flag is True

            assert isinstance(wts_n_iters, int)
            assert wts_n_iters >= 1

        self._sett_wts_obj_wts = weights
        self._sett_wts_obj_auto_set_flag = auto_wts_set_flag
        self._sett_wts_obj_n_iters = wts_n_iters

        if self._vb:
            print(
                'Automatic detection of weights:',
                self._sett_wts_obj_auto_set_flag)

            if not self._sett_wts_obj_auto_set_flag:
                print(
                    'Objective function weights:',
                    self._sett_wts_obj_wts)

            else:

                print('Iterations to estimate weights:',
                    self._sett_wts_obj_n_iters)

            print_el()

        self._sett_wts_obj_set_flag = True
        return

    def set_selective_phsann_settings(self, min_period, max_period):

        '''
        Phase anneal some phases only.

        Phases having periods less than min_period and greater than max period
        are left untouched.

        Parameters:
        ----------
        min_period : int or None
            Phases having periods less than min_period are not
            annealed/randomized. Should be greater than zero and less than
            max_period. An error is raised if min_period does not exist in the
            data.
        max_period : int or None
            Phases having periods greater than max_period are not
            annealed/randomized. Should be greater than zero and greater than
            min_period. An error is raised if max_period does not exist in the
            data.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting selective phases for phase annealing...\n')

        if isinstance(min_period, int):
            assert min_period > 0, 'Invalid min_period!'

        elif min_period is None:
            pass

        else:
            raise AssertionError('min_period can only be None or an int!')

        if isinstance(max_period, int):
            assert max_period > 0, 'Invalid max_period!'

        elif max_period is None:
            pass

        else:
            raise AssertionError('max_period can only be None or an int!')

        if isinstance(min_period, int) and isinstance(max_period, int):
            assert max_period > min_period, (
                'max_period must be greater than min_period!')

        self._sett_sel_phs_min_prd = min_period
        self._sett_sel_phs_max_prd = max_period

        if self._vb:
            print('Minimum period:', self._sett_sel_phs_min_prd)
            print('Maximum period:', self._sett_sel_phs_max_prd)

            print_el()

        self._sett_sel_phs_set_flag = True
        return

    def set_lags_nths_weights_settings(
            self,
            lags_nths_exp,
            lags_nths_n_iters,
            lags_nths_cumm_wts_contrib,
            lags_nths_n_thresh):

        '''
        Individual lag and nth order weights for each objective function.

        For cases such as discharge where asymmetries are highly non-normal,
        Simulated Annealing ignores the lags with high asymmetries.

        By assigning more weights to steps with higher differences w.r.t
        reference, the aforementioned problem can be solved.

        Only works if distribution fitting is on, otherwise an error is
        raised during verification. The weights are estimated by calling the
        objective function repeatedly with random phase changes.

        The weights are distributed such that the final objective function
        is same as that without the weights. The difference happens for the
        behaviour of the objective functions after the application of weights.
        i.e. They may behave more erratic if some lag/nths have much higher
        errors than the rest.

        At least one of the specified lags or nth orders should be > 1 i.e.
        both cannot have lengths of 1.

        Lags or nth orders can be selected automatically by cumm_wts_contrib
        or n_thresh. This increases the speed of computing the
        objective function by evaluating the objective function at the
        relevent points only.

        After computing weights for each lag or nth order, lags and
        nth orders that have very little weights can be turned off
        for objective function evaluation using cumm_wts_contrib and/or
        n_thresh.

        Parameters
        ----------
        lags_nths_exp : int or float
            An exponent to scale the weights at each lag/nth order.
            Higher means more weight at lags/nth orders that have more error.
            This is done for each variable independently. Should be >= 1
            and < infinity.
        lags_nths_n_iters : int
            Number of iterations to estimate the weights. Should be greater
            than 0.
        lags_nths_cumm_wts_contrib : float or None
            Relative cummulative weights' limit such that all the sorted
            weights for a given series contribute almost
            lags_nths_cumm_wts_contrib amount to the total sum of the
            weights. For example, if lags_nths_cumm_wts_contrib is 0.9
            then all lags or nth orders that contribute 0.9 to the weights
            are taken into account. The rest are set to zero. The algorithm is
            applied after the weights are computed using lags_nths_exp.
            Finally, the minimum number of lags or nth orders produced by
            lags_nths_n_thresh and lags_nths_n_thresh is used.
            If None, then not used. If float then should be > zero and <= 1.
        lags_nths_n_thresh : integer or None
            Take only the lags or nth orders that contribute the biggest
            N errors for a given series. Finally, the minimum number of lags
            or nth orders produced by lags_nths_n_thresh and
            lags_nths_n_thresh is used.
            If None, then not used. Else should be >= 1 <= max. of lags or
            nth orders.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting lags\' and nths\' weights settings for phase '
                'annealing...\n')

        assert isinstance(lags_nths_exp, (int, float)), (
            'lags_nths_exp not and integer or float!')

        lags_nths_exp = float(lags_nths_exp)

        assert 1 <= lags_nths_exp <= np.inf, 'Invalid lags_nths_exp!'

        assert isinstance(lags_nths_n_iters, int), (
            'lags_nths_n_iters not an integer!')

        assert lags_nths_n_iters > 0, 'Invalid lags_nths_n_iters!'

        if lags_nths_cumm_wts_contrib is not None:
            assert isinstance(lags_nths_cumm_wts_contrib, float), (
                'lags_nths_cumm_wts_contrib not a float!')

            assert 0 < lags_nths_cumm_wts_contrib <= 1, (
                'Invalid lags_nths_cumm_wts_contrib!')

        if lags_nths_n_thresh is not None:
            assert isinstance(lags_nths_n_thresh, int), (
                'lags_nths_n_thresh not an integer!')

            assert lags_nths_n_thresh >= 1

        self._sett_wts_lags_nths_exp = lags_nths_exp
        self._sett_wts_lags_nths_n_iters = lags_nths_n_iters
        self._sett_wts_lags_nths_cumm_wts_contrib = lags_nths_cumm_wts_contrib
        self._sett_wts_lags_nths_n_thresh = lags_nths_n_thresh

        if self._vb:
            print(
                'Lags\' and nths\' exponent:',
                self._sett_wts_lags_nths_exp)

            print(
                'Iteration to estimate weights:',
                self._sett_wts_lags_nths_n_iters)

            print(
                'Cummulative weights\' contribution threshold:',
                self._sett_wts_lags_nths_cumm_wts_contrib)

            print(
                'Maximum lags or nth orders\' threshold:',
                self._sett_wts_lags_nths_n_thresh)

            print_el()

        self._sett_wts_lags_nths_set_flag = True
        return

    def set_label_weights_settings(self, label_exp, label_n_iters):

        '''
        Individual label weights for each objective function.

        Similar to set_lags_nths_weights_settings, but for labels/columns.

        For multivariate simulations, some variables might get little
        consideration in the objective function.

        By assigning more weights to labels with higher differences w.r.t
        reference, the aforementioned problem can be solved.

        Only works if distribution fitting is on, otherwise an error is
        raised during verification. The weights are estimated by calling the
        objective function repeatedly, before the algorithm begins, with
        random phase changes and after lags and nth weights are computed.

        The weights are distributed such that the final objective function
        is same as that without the weights. The difference happens for the
        behaviour of the objective functions after the application of weights.
        i.e. They may behave more erratic values if some labels have much
        higher errors than the rest.

        Parameters
        ----------
        label_exp : int or float
            An exponent to scale the weights for each label.
            Higher means more weight at label that have more error.
            This is done for each variable independently. Should be >= 1
            and < infinity.
        label_n_iters : int
            Number of iterations to estimate the weights. Should be greater
            than 0.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting label weights\' settings for phase '
                'annealing...\n')

        assert isinstance(label_exp, (int, float)), (
            'label_exp not and integer or float!')

        label_exp = float(label_exp)

        assert 1 <= label_exp <= np.inf, 'Invalid label_exp!'

        assert isinstance(label_n_iters, int), (
            'label_n_iters not an integer!')

        assert label_n_iters > 0, 'Invalid label_n_iters!'

        self._sett_wts_label_exp = label_exp
        self._sett_wts_label_n_iters = label_n_iters

        if self._vb:
            print(
                'Lags\' and nths\' exponent:',
                self._sett_wts_label_exp)

            print(
                'Iteration to estimate weights:',
                self._sett_wts_label_n_iters)

            print_el()

        self._sett_wts_label_set_flag = True
        return

    def set_cdf_penalty_settings(self, n_vals_thresh, n_vals_penlt):

        '''
        Set penalty for simulated CDF values in case they deviate from a
        threshold region of the reference. It is only applicable where
        CDFs are used.

        Parameters
        ----------
        n_vals_thresh : integer
            The number of simulated CDF values that are considered to have
            an acceptable deviation from the reference. For example,
            We have 10 values in the reference CDF. For a given reference,
            CDF value, say, 0.7 and n_vals_thresh of 1 would mean that for
            the simulated value's probability value with a +- 1/10 of 0.7 i.e.
            between 0.6 and 0.8, a penalty is not applied. It should be
            greater than zero.
        n_vals_penlt : integer
            How much penalty to apply if simulated values are out of the
            threshold limits. In the aforementioned case, if the simulated
            value's CDF value is, say, 0.5 then a penalty of n_vals_penlt/10
            is applied to the value by adding n_vals_penlt/10 to the deviation.
            Should be greater than zero and n_vals_thresh. The sign of the
            penalty is taken care of accordingly.
        '''

        if self._vb:
            print_sl()

            print('Setting CDF penalties settings for phase annealing...\n')

        assert isinstance(n_vals_thresh, int), 'n_vals_thresh not an integer!'
        assert isinstance(n_vals_penlt, int), 'n_vals_penlt not an integer!'

        assert n_vals_thresh > 0, 'Invalid n_vals_thresh!'
        assert n_vals_penlt > 0, 'Invalid n_vals_penlt!'

        assert n_vals_penlt > n_vals_thresh, (
            'n_vals_penlt should be > n_vals_thresh!')

        self._sett_cdf_pnlt_n_thrsh = n_vals_thresh
        self._sett_cdf_pnlt_n_pnlt = n_vals_penlt

        if self._vb:
            print(
                'Set number of threshold values for penalty:',
                self._sett_cdf_pnlt_n_thrsh)

            print('Set number of penalty values:',
                self._sett_cdf_pnlt_n_pnlt)

            print_el()

        self._sett_cdf_pnlt_set_flag = True
        return

    def set_partial_cdf_calibration_settings(
            self, lower_threshold, upper_threshold, inside_flag):

        '''
        Set limits on CDFs from or within which deviations affect the final
        objective function value.

        Parameters
        ----------
        lower_threshold : float or None
            Lower threshold for CDFs below or above which to take the
            deviations from the reference. Can be > 0 and < 1 or None.
            Should be > upper_threshold if upper_threshold is a float.
            If None, then this parameter is not considered. At least
            one of lower_threshold and upper_threshold must be a float.
            The way lower_threshold is used, is based on inside_flag.
            All the above is True for functions whose mean is around zero e.g.
            Asymmetry and Nth Order differences. For other distributions e.g.
            Correlations or Entropy, sum of lower_threshold and
            upper_threshold is used along with the state of inside_flag to
            account for the proper values.
        upper_threshold: float or None
            Upper threshold for CDFs below or above which to take the
            deviations from the reference. Can be > 0 and < 1 or None.
            Should be > than lower_threshold if lower_threshold is a float.
            If None, then this parameter is not considered. At least
            one of lower_threshold and upper_threshold must be a float.
            The way upper_threshold is used, is based on inside_flag.
            All the above is True for functions whose mean is around zero e.g.
            Asymmetry and Nth Order differences. For other distributions e.g.
            Correlations or Entropy, sum of lower_threshold and
            upper_threshold is used along with the state of inside_flag to
            account for the proper values.
        inside_flag : bool
            Set the side of CDF values to take into account using the
            limits lower_threshold and upper_threshold. If True, then
            CDF values >= lower_threshold and <= upper_threshold are taken
            into account, given both are not None. If False, then CDF values
            <= lower_threshold and >= upper_threshold are taken into account,
            provided both are not None. If True and lower_threshold is None,
            then all values below upper_threshold are taken into account.
            In other words, inside_flag allows to take values from
            lower_threshold or upper_threshold towards zero. This zero can be
            at the center (Gaussian-type distribution) or on one side
            (exponential-type distributions).
        '''

        if self._vb:
            print_sl()

            print(
                'Setting partial CDF calibration settings for phase '
                'annealing...\n')

        lt_n = lower_threshold is not None
        ut_n = upper_threshold is not None

        if lt_n:
            assert isinstance(lower_threshold, float), (
                'lower_threshold not a float!')

            assert 0 < lower_threshold < 1, 'Invalid lower_threshold!'

        if ut_n:
            assert isinstance(upper_threshold, float), (
                'upper_threshold not a float!')

            assert 0 < upper_threshold < 1, 'Invalid upper_threshold!'

        if lt_n and ut_n:
            assert lower_threshold < upper_threshold, (
                'lower_threshold must be smaller than upper_threshold!')

        assert any([lt_n, ut_n]), (
            'At least one of lower_threshold and upper_threshold must be a'
            'float!')

        assert isinstance(inside_flag, bool), (
            'inside_flag not a boolean!')

        if lt_n:
            self._sett_prt_cdf_calib_lt = lower_threshold

        if ut_n:
            self._sett_prt_cdf_calib_ut = upper_threshold

        self._sett_prt_cdf_calib_inside_flag = inside_flag

        if self._vb:
            print(
                'Set lower threshold to:',
                self._sett_prt_cdf_calib_lt)

            print(
                'Set upper threshold to:',
                self._sett_prt_cdf_calib_ut)

            print(
                'Set inside flag to:',
                self._sett_prt_cdf_calib_inside_flag)

            print_el()

        self._sett_prt_cdf_calib_set_flag = True
        return

    def set_misc_settings(self, n_rltzns, outputs_dir, n_cpus):

        '''
        Some more parameters

        Parameters
        ----------
        n_rltzns : integer
            The number of realizations to generate. Should be > 0
        outputs_dir : str, Path-like
            Path to the directory where the outputs will be stored.
            Created if not there.
        n_cpus : string, integer
            Maximum number of processes to use to generate realizations.
            If the string 'auto' then the number of logical cores - 1
            processes are used. If an integer > 0 then that number of
            processes are used.
        '''

        if self._vb:
            print_sl()

            print('Setting misc. settings for phase annealing...\n')

        assert isinstance(n_rltzns, int), 'n_rltzns not an integer!'
        assert 0 < n_rltzns, 'Invalid n_rltzns!'

        outputs_dir = Path(outputs_dir)

        assert outputs_dir.is_absolute()

        assert outputs_dir.parents[0].exists(), (
            'Parent directory of outputs dir does not exist!')

        if not outputs_dir.exists:
            outputs_dir.mkdir(exist_ok=True)

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'Invalid n_cpus!'

            n_cpus = get_n_cpus()

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        self._sett_misc_n_rltzns = n_rltzns
        self._sett_misc_outs_dir = outputs_dir
        self._sett_misc_n_cpus = n_cpus

        if self._vb:
            print(
                'Number of realizations:',
                self._sett_misc_n_rltzns)

            print(
                'Outputs directory:',
                self._sett_misc_outs_dir)

            print(
                'Number of maximum process(es) to use:',
                self._sett_misc_n_cpus)

            print_el()

        self._sett_misc_set_flag = True
        return

    def verify(self):

        PAD._PhaseAnnealingData__verify(self)
        assert self._data_verify_flag, 'Data in an unverified state!'

        assert self._sett_obj_set_flag, 'Call set_objective_settings first!'
        assert self._sett_ann_set_flag, 'Call set_annealing_settings first!'
        assert self._sett_misc_set_flag, 'Call set_misc_settings first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Algorithm meant for 2D only!')

        assert np.all(
            self._sett_obj_lag_steps < self._data_ref_shape[0]), (
                'At least one of the lag_steps is >= the size '
                'of ref_data!')

        assert np.all(
            self._sett_obj_lag_steps_vld < self._data_ref_shape[0]), (
                'At least one of the lag_steps_vld is >= the size '
                'of ref_data!')

        assert self._sett_obj_ecop_dens_bins <= self._data_ref_shape[0], (
           'ecop_dens_bins have to be <= size of ref_data!')

        assert self._sett_obj_nth_ords.max() < self._data_ref_shape[0], (
            'Maximum of nth_ord is >= the size of ref_data!')

        assert self._sett_obj_nth_ords_vld.max() < self._data_ref_shape[0], (
            'Maximum of nth_ord_vld is >= the size of ref_data!')

        if any([self._sett_obj_asymm_type_1_ms_flag,
                self._sett_obj_asymm_type_2_ms_flag,
                self._sett_obj_ecop_dens_ms_flag,
                self._sett_obj_asymm_type_1_ms_ft_flag,
                self._sett_obj_asymm_type_2_ms_ft_flag,
                self._sett_obj_etpy_ms_ft_flag]):

            assert self._data_ref_n_labels > 1, (
                'More than one time series needed for multisite asymmetries!')

        if self._sett_auto_temp_set_flag:
            self._sett_misc_auto_init_temp_dir = (
                self._sett_misc_outs_dir / 'auto_init_temps__acpt_rates')

            self._sett_ann_init_temp = None

            if self._vb:
                print_sl()

                print(
                    'Set Phase annealing initial temperature to None due '
                    'to auto search!')

                print_el()

        self._sett_obj_flag_vals = np.array([
            self._sett_obj_scorr_flag,
            self._sett_obj_asymm_type_1_flag,
            self._sett_obj_asymm_type_2_flag,
            self._sett_obj_ecop_dens_flag,
            self._sett_obj_ecop_etpy_flag,
            self._sett_obj_nth_ord_diffs_flag,
            self._sett_obj_cos_sin_dist_flag,
            self._sett_obj_pcorr_flag,
            self._sett_obj_asymm_type_1_ms_flag,
            self._sett_obj_asymm_type_2_ms_flag,
            self._sett_obj_ecop_dens_ms_flag,
            self._sett_obj_match_data_ft_flag,
            self._sett_obj_match_probs_ft_flag,
            self._sett_obj_asymm_type_1_ft_flag,
            self._sett_obj_asymm_type_2_ft_flag,
            self._sett_obj_nth_ord_diffs_ft_flag,
            self._sett_obj_asymm_type_1_ms_ft_flag,
            self._sett_obj_asymm_type_2_ms_ft_flag,
            self._sett_obj_etpy_ft_flag,
            self._sett_obj_etpy_ms_ft_flag,
            ])

        assert (self._sett_obj_flag_labels.size ==
                self._sett_obj_flag_vals.size), (
                    'Number of objective function flags\' labels and '
                    'values do not correspond!')

        if self._sett_wts_obj_set_flag and self._sett_wts_obj_wts is not None:
            self._sett_wts_obj_wts = self._sett_wts_obj_wts[
                self._sett_obj_flag_vals].copy()

            assert np.all(self._sett_wts_obj_wts != 0), (
                'At least one objective function that is on has a weight of '
                'zero!')

        if self._sett_wts_obj_set_flag:
            assert self._sett_obj_flag_vals.sum() > 1, (
                'At least two objective function flag must be True for '
                'objective weights to be applied!')

        self._sett_wts_lags_obj_flags = [
                self._sett_obj_scorr_flag,
                self._sett_obj_asymm_type_1_flag,
                self._sett_obj_asymm_type_2_flag,
                self._sett_obj_ecop_dens_flag,
                self._sett_obj_ecop_etpy_flag,
                self._sett_obj_pcorr_flag,
                self._sett_obj_asymm_type_1_ft_flag,
                self._sett_obj_asymm_type_2_ft_flag,
                self._sett_obj_etpy_ft_flag]

        self._sett_wts_nths_obj_flags = [
                self._sett_obj_nth_ord_diffs_flag,
                self._sett_obj_nth_ord_diffs_ft_flag]

        if self._sett_wts_lags_nths_set_flag:
            assert self._sett_obj_use_obj_dist_flag, (
                'Distribution fitting flag must be True for lags\' and '
                'nths\' weights!')

            assert (any(self._sett_wts_lags_obj_flags) or
                    any(self._sett_wts_nths_obj_flags)), (
                'None of the objective function flags related to lags and '
                'nths weights computation are active!')

            if any(self._sett_wts_lags_obj_flags):
                if any(self._sett_wts_nths_obj_flags):
                    assert (
                        (self._sett_obj_lag_steps.size > 1) or
                        (self._sett_obj_nth_ords.size > 1)), (
                        'More than one lag or nth order required to '
                        'compute lag or nth order weights!')

                    if self._sett_wts_lags_nths_n_thresh is not None:
                        assert (
                           (self._sett_wts_lags_nths_n_thresh <=
                            self._sett_obj_lag_steps.size
                           ) or (
                            self._sett_wts_lags_nths_n_thresh <=
                            self._sett_obj_nth_ords.size)), (
                            'lags_nths_n_thresh greater than the number '
                            'of lags and nth orders!')

                else:
                    assert self._sett_obj_lag_steps.size > 1, (
                        'More than one lag required to compute lag weights!')

                    if self._sett_wts_lags_nths_n_thresh is not None:
                        assert (
                           self._sett_wts_lags_nths_n_thresh <=
                           self._sett_obj_lag_steps.size), (
                           'lags_nths_n_thresh greater than the number '
                           'of lags!')

            elif any(self._sett_wts_nths_obj_flags):
                assert self._sett_obj_nth_ords.size > 1, (
                    'More than one nth order required to compute nth order '
                    'weights!')

                if self._sett_wts_lags_nths_n_thresh is not None:
                    assert (
                        self._sett_wts_lags_nths_n_thresh <=
                        self._sett_obj_nth_ords.size), (
                        'lags_nths_n_thresh greater than the number of nth '
                        'orders!')

        if self._sett_wts_label_set_flag:
            assert self._sett_obj_use_obj_dist_flag, (
                'Distribution fitting flag must be True for label weights!')

            assert self._data_ref_n_labels > 1, (
                'More than one label required for label weights!')

        if self._sett_cdf_pnlt_set_flag:
            assert self._sett_obj_use_obj_dist_flag, (
                'For using CDF penalties, the flag to use distributions in '
                'the objective functions must be turned on!')

        if self._sett_prt_cdf_calib_set_flag:
            assert self._sett_obj_use_obj_dist_flag, (
                'For using partial CDF calibration, the flag to use '
                'distributions in the objective functions must be turned on!')

        if self._sett_obj_use_dens_ftn_flag:
            n_vals_per_bin = int(
                self._sett_obj_ratio_per_dens_bin *
                (self._data_ref_shape[0] -
                 max(self._sett_obj_nth_ords_vld.max(),
                     self._sett_obj_lag_steps_vld.max())))

            # For ecop density functions, this condition is not fulfilled.
            # There, a minimum of two values are taken to get the bins.
            assert n_vals_per_bin > 1, (
                    'ratio_per_dens_bin is too '
                    'small for the given number of data points!')

            print(
                f'At maximum {n_vals_per_bin} values per bin in the '
                f'empirical density function.')

        if self._vb:
            print_sl()

            print(f'Phase annealing settings verified successfully!')

            print_el()

        self._sett_verify_flag = True
        return

    __verify = verify
