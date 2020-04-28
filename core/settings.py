'''
Created on Dec 27, 2019

@author: Faizan
'''
from pathlib import Path

import psutil
import numpy as np

from ..misc import print_sl, print_el

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
        # class.
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
        self._sett_obj_pcorr_flag = None
        self._sett_obj_lag_steps_vld = None
        self._sett_obj_nth_ords_vld = None
        self._sett_obj_asymm_type_1_ms_flag = None
        self._sett_obj_asymm_type_2_ms_flag = None
        self._sett_obj_n_flags = 11

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
        self._sett_ann_phs_ann_class_width = None

        # Automatic initialization temperature.
        self._sett_ann_auto_init_temp_temp_bd_lo = None
        self._sett_ann_auto_init_temp_temp_bd_hi = None
        self._sett_ann_auto_init_temp_atpts = None
        self._sett_ann_auto_init_temp_niters = None
        self._sett_ann_auto_init_temp_acpt_bd_lo = None
        self._sett_ann_auto_init_temp_acpt_bd_hi = None
        self._sett_ann_auto_init_temp_trgt_acpt_rate = None
        self._sett_ann_auto_init_temp_ramp_rate = None
        self._sett_ann_auto_init_temp_n_rltzns = None

        # Extended length.
        # Internally using shape instead of scalar length.
        self._sett_extnd_len_rel_shp = np.array([1, ], dtype=int)

        # Multiple phase annealing.
        self._sett_mult_phs_n_beg_phss = 1
        self._sett_mult_phs_n_end_phss = 1

        # Misc.
        self._sett_misc_n_rltzns = None
        self._sett_misc_outs_dir = None
        self._sett_misc_n_cpus = None

        # Flags.
        self._sett_obj_set_flag = False
        self._sett_ann_set_flag = False
        self._sett_auto_temp_set_flag = False
        self._sett_extnd_len_set_flag = False
        self._sett_mult_phs_flag = False
        self._sett_misc_set_flag = False

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
            asymm_type_2_ms_flag):

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
            All should be greater than zero and unique. This parameter is
            needed when any of the scorr_flag, asymm_type_1_flag,
            asymm_type_2_flag, ecop_dens_flag are True.
        ecop_dens_bins : integer
            Number of bins for the empirical copula. The more the bins the
            finer the density match between the reference and simulated.
            This parameter is required when the ecop_dens_flag is True.
        nth_ords : 1D integer np.ndarray
            Order of differences (1st, 2nd, ...) if nth_order_diffs_flag
            is True.
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
        nth_ords_vld : 1D integer np.ndarray
            Same as nth_ords but these orders are used for plots. Computed
            at the end of simulation. Also, it is unionized with nth_ords.
        asymm_type_1_ms_flag : bool
            Whether to minimize the differences between the first type
            normalized asymmetry of the reference and generated realizations,
            for multisite copulas.
        asymm_type_2_ms_flag : bool
            Whether to minimize the differences between the second type
            normalized asymmetry of the reference and generated realizations,
            for multisite copulas.
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
            asymm_type_2_ms_flag
            ]), 'All objective function flags are False!'

        assert isinstance(lag_steps, np.ndarray), (
            'lag_steps not a numpy arrray!')

        assert lag_steps.ndim == 1, 'lag_steps not a 1D array!'
        assert lag_steps.size > 0, 'lag_steps is empty!'
        assert lag_steps.dtype == np.int, 'lag_steps has a non-integer dtype!'
        assert np.all(lag_steps > 0), 'lag_steps has non-postive values!'

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

        assert np.unique(nth_ords_vld).size == nth_ords_vld.size, (
            'Non-unique values in nth_ords_vld!')

        assert nth_ords_vld.max() < 1000, (
            'Maximum of nth_ords_vld cannot be more than 1000!')

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
        self._sett_obj_pcorr_flag = pcorr_flag
        self._sett_obj_asymm_type_1_ms_flag = asymm_type_1_ms_flag
        self._sett_obj_asymm_type_2_ms_flag = asymm_type_2_ms_flag

        self._sett_obj_lag_steps_vld = np.sort(lag_steps_vld).astype(np.int64)

        self._sett_obj_lag_steps_vld = np.union1d(
            self._sett_obj_lag_steps, self._sett_obj_lag_steps_vld)

        self._sett_obj_lag_steps_vld = np.sort(self._sett_obj_lag_steps_vld)

        self._sett_obj_nth_ords_vld = np.sort(nth_ords_vld).astype(np.int64)

        self._sett_obj_nth_ords_vld = np.union1d(
            self._sett_obj_nth_ords, self._sett_obj_nth_ords_vld)

        self._sett_obj_nth_ords_vld = np.sort(self._sett_obj_nth_ords_vld)

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
            phase_annealing_class_width):

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
            3:    Reduction rate is equal to the mean accepatance rate of
                  previous acceptance_rate_iterations.
        mag_spec_index_sample_flag : bool
            Whether to sample new freqeuncy indices on a magnitude spectrum
            CDF based weighting i.e. frequencies having more amplitude
            have a bigger chance of being sampled.
        phase_reduction_rate : float
            If phase_reduction_rate_type is 2, then the new phase reduction
            rate is previous multiplied by phase_reduction_rate_type. Should
            be > 0 and <= 1.
        phase_annealing_class_width : int
            Whether to anneal the phases in given class width. In the inverse
            FT, the power spectrum for is also used till the end of the
            current class. e.g. there are 11 phases and class width is 2. In
            the first attempt, phases are optimized for the inverse FT that
            has only two coeffs. The rest are zero. The  second attempt has
            4 FT coefficients but phases of optimized for the last two only.
            The final classs will have only one phase for optimization.
            Should be greater than zero. If more than the total number of
            coefficients, then all phases are optimized and only one attempt
            is made (the case with 1 class only). Every class is optimized
            till acpt_rate drops to stop_acpt_rate or less. Then optimization
            for the next class starts anew and so on till all classes are
            optimized.
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

        assert isinstance(phase_annealing_class_width, (int, None)), (
            'phase_annealing_class_width not an integer or None!')

        if phase_annealing_class_width is not None:
            assert phase_annealing_class_width > 0, (
                'Invalid phase_annealing_class_width!')

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
        self._sett_ann_phs_ann_class_width = phase_annealing_class_width

        if phase_reduction_rate_type == 2:
            self._sett_ann_phs_red_rate = phase_reduction_rate

        elif phase_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError('Unknown phase_reduction_rate_type!')

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
                'Initial phase annealing class width:',
                self._sett_ann_phs_ann_class_width)

            print_el()

        self._sett_ann_set_flag = True
        return

    def set_annealing_auto_temperature_settings(
            self,
            temperature_lower_bound,
            temperature_upper_bound,
            max_search_attempts,
            n_iterations_per_attempt,
            acceptance_lower_bound,
            acceptance_upper_bound,
            target_acceptance_rate,
            ramp_rate):

        '''
        Automatic annealing initial temperature search parameters. Each
        realization will get its own initial temperature. It is sampled
        on a uniform interval between the temperatures that correspond to
        the minimum and maximum acceptance rate approximately.

        Parameters
        ----------
        temperature_lower_bound : float
            Lower bound of the temperature search space. Should be > 0
        temperature_upper_bound : float
            Upper bound of the temperature search space. Should be >
            temperature_lower_bound and < infinity.
        max_search_attempts : integer
            Maximum number of attempts to search for the temperature if
            no other stopping criteria are met. Should be > 0.
        n_iterations_per_attempt : integer
            Number of times to run the annealing algorithm after which to
            compute the mean acceptance rate. Should be large enough to
            give a stable acceptance rate. Should be > 0.
        acceptance_lower_bound : float
            Lower bounds of the acceptance rate for an initial starting
            temperature to be accepted for the optimization. Should be >= 0
            and < 1.
        acceptance_upper_bound : float
            Upper bounds of the acceptance rate for an initial starting
            temperature to be accepted for the optimization. Should be >
            acceptance_lower_bound and < 1.
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

        assert isinstance(max_search_attempts, int), (
            'max_search_attempts not an integer!')

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

        assert 0 < max_search_attempts, 'Invalid max_search_attempts!'

        assert 0 < n_iterations_per_attempt, (
            'Invalid n_iterations_per_attempt!')

        assert (
            0 <
            acceptance_lower_bound <=
            target_acceptance_rate <=
            acceptance_upper_bound <
            1.0), (
                'Invalid or inconsistent acceptance_lower_bound, '
                'target_acceptance_rate or acceptance_upper_bound!')

        assert 1 < ramp_rate < np.inf, 'Invalid ramp_rate!'

        self._sett_ann_auto_init_temp_temp_bd_lo = temperature_lower_bound
        self._sett_ann_auto_init_temp_temp_bd_hi = temperature_upper_bound
        self._sett_ann_auto_init_temp_atpts = max_search_attempts
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
                'Maximum temperature search attempts:',
                self._sett_ann_auto_init_temp_atpts)

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

    def set_extended_length_sim_settings(self, relative_length):

        '''
        Parameters for simulating series longer than reference data length

        Parameters
        ----------
        relative_length: integer
            Relative length of the simulated series. Should be >=
            1. e.g. if reference has 100 steps and
            relative_length is 4 then the length of the simulated series
            is 400 steps.
        '''

        if self._vb:
            print_sl()

            print(
                'Setting multi-length simulation settings for '
                'phase annealing...\n')

        assert isinstance(relative_length, int), (
            'relative_length not an integer!')

        assert relative_length >= 1, 'Invalid relative_length!'

        # made for multidimensions (to come later)
        self._sett_extnd_len_rel_shp = np.array(
            [relative_length], dtype=int)

        if self._vb:
            print('Relative length:', self._sett_extnd_len_rel_shp[0])

            print_el()

        self._sett_extnd_len_set_flag = True
        return

    def set_mult_phase_settings(self, n_beg_phss, n_end_phss):

        '''
        Randomize multiple phases instead of just one.

        A random number of phases are generated for each iteration between
        n_beg_phss and n_end_phss (both inclusive). These values are reduced
        if available phases/magnitudes are not enough.

        Parameters
        ----------
        n_beg_phss : integer
            Minimum phases/magnitudes to randomize per iteration.
            Should be > 0.
        n_end_phss : integer
            Maximum number of phases/magnitudes to randomize per iteration.
            Should be >= n_beg_phss.
        '''

        if self._vb:
            print_sl()

            print('Settings multiple phase annealing settings...\n')

        assert isinstance(n_beg_phss, int), 'n_beg_phss not an integer!'
        assert isinstance(n_end_phss, int), 'n_end_phss not an integer!'

        assert n_beg_phss > 0, 'Invalid n_beg_phss!'
        assert n_end_phss >= n_beg_phss, 'Invalid n_end_phss!'

        self._sett_mult_phs_n_beg_phss = n_beg_phss
        self._sett_mult_phs_n_end_phss = n_end_phss

        if self._vb:
            print(
                f'Starting multiple phase indices: '
                f'{self._sett_mult_phs_n_beg_phss}')

            print(
                f'Ending multiple phase indices: '
                f'{self._sett_mult_phs_n_end_phss}')

            print_el()

        self._sett_mult_phs_flag = True
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

            n_cpus = max(1, psutil.cpu_count() - 1)

        else:
            assert isinstance(n_cpus, int), 'n_cpus is not an integer!'

            assert n_cpus > 0, 'Invalid n_cpus!'

        if n_rltzns < n_cpus:
            n_cpus = n_rltzns

        self._sett_misc_n_rltzns = n_rltzns
        self._sett_misc_outs_dir = outputs_dir
        self._sett_misc_n_cpus = n_cpus

        if self._vb:
            print('Number of realizations:', self._sett_misc_n_rltzns)

            print('Outputs directory:', self._sett_misc_outs_dir)

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
                self._sett_obj_asymm_type_2_ms_flag]):

            assert self._data_ref_n_labels > 1, (
                'More than one time series needed for multisite asymmetries!')

        if self._sett_auto_temp_set_flag:
            self._sett_ann_init_temp = None

            if self._vb:
                print_sl()

                print(
                    'Set Phase annealing initial temperature to None due '
                    'to auto search!')

                print_el()

        if self._sett_extnd_len_rel_shp[0] > 1:
            if self._sett_obj_nth_ord_diffs_flag:
                raise NotImplementedError('Don\'t know how to do this yet!')

            if self._sett_obj_cos_sin_dist_flag:
                raise NotImplementedError('Don\'t know how to do this yet!')

        if self._vb:
            print_sl()

            print(f'Phase annealing settings verified successfully!')

            print_el()

        self._sett_verify_flag = True
        return

    __verify = verify
