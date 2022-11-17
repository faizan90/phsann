'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from gnrctsgenr import GTGSettings

from gnrctsgenr.misc import print_sl, print_el


class PhaseAnnealingSettings(GTGSettings):

    def __init__(self):

        GTGSettings.__init__(self)

        # Simulated Annealing.
        self._sett_ann_phs_red_rate_type = None
        self._sett_ann_mag_spec_cdf_idxs_flag = None
        self._sett_ann_phs_red_rate = None
        self._sett_ann_min_phs_red_rate = None

        # Multiple phase annealing.
        self._sett_mult_phs_n_beg_phss = None
        self._sett_mult_phs_n_end_phss = None
        self._sett_mult_phs_sample_type = None
        self._sett_mult_phss_red_rate = None

        # Selective phase annealing.
        self._sett_sel_phs_min_prd = None
        self._sett_sel_phs_max_prd = None
        self._sett_sel_phs_beyond_flag = None

        # Initial phase spectra.
        self._sett_init_phs_spec_type = None
        self._sett_init_phs_specs = None

        # Limited phase randomization.
        self._sett_lim_phsrand_obj_lbd = None
        self._sett_lim_phsrand_obj_ubd = None
        self._sett_lim_phsrand_ptrb_lbd = None
        self._sett_lim_phsrand_ptrb_ubd = None
        self._sett_lim_phsrand_n_ptrb_vals = None
        self._sett_lim_phsrand_iters_per_atpt = None

        # Flags.
        self._sett_ann_pa_sa_sett_flag = False
        self._sett_mult_phs_flag = False
        self._sett_sel_phs_set_flag = False
        self._sett_init_phs_spec_set_flag = False
        self._sett_lim_phsrand_set_flag = False
        self._sett_ann_pa_sa_sett_verify_flag = False
        return

    def set_pa_sa_settings(
            self,
            phase_reduction_rate_type,
            mag_spec_index_sample_flag,
            phase_reduction_rate,
            min_phs_red_rate):

        '''
        Addtional variables used during annealing realted to phase annealing.

        Parameters
        ----------
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
        min_phs_red_rate : float
            The minimum phase reduction rate, below which the change is
            considered as zero. Must be greater than or equal to zero
            and less than one.
        '''

        if self._vb:
            print_sl()

            print('Setting additonal phase annealing parameters...\n')

        assert isinstance(phase_reduction_rate_type, int), (
            'phase_reduction_rate_type not an integer!')

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

        assert isinstance(min_phs_red_rate, float), (
            'min_phs_red_rate not a float!')

        assert 0 <= min_phs_red_rate < 1.0, (
            'Invalid min_phs_red_rate!')

        self._sett_ann_phs_red_rate_type = phase_reduction_rate_type

        self._sett_ann_mag_spec_cdf_idxs_flag = mag_spec_index_sample_flag

        if phase_reduction_rate_type == 2:
            self._sett_ann_phs_red_rate = phase_reduction_rate

        elif phase_reduction_rate_type in (0, 1, 3):
            pass

        else:
            raise NotImplementedError('Unknown phase_reduction_rate_type!')

        self._sett_ann_min_phs_red_rate = min_phs_red_rate

        if self._vb:

            print(
                'Phase reduction rate type:',
                self._sett_ann_phs_red_rate_type)

            print(
                'Magnitude spectrum based indexing flag:',
                self._sett_ann_mag_spec_cdf_idxs_flag)

            print(
                'Phase reduction rate:', self._sett_ann_phs_red_rate)

            print(
                'Minimum phase reduction rate:',
                self._sett_ann_min_phs_red_rate)

            print_el()

        self._sett_ann_pa_sa_sett_flag = True
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

            print('Setting multiple phase annealing parameters...\n')

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

    def set_selective_phsann_settings(
            self,
            min_period,
            max_period,
            keep_beyond_flag):

        '''
        Phase anneal some phases only.

        Parameters:
        ----------
        min_period : int or None
            The lower range of the period.
        max_period : int or None
            The upper range of the period.
        keep_beyond_flag : bool
            Whether to keep the phases on the inside or the outside of
            minperiod or/and max_period.
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

        assert isinstance(keep_beyond_flag, bool), (
            'keep_beyond_flag can only be a boolean!')

        self._sett_sel_phs_min_prd = min_period
        self._sett_sel_phs_max_prd = max_period
        self._sett_sel_phs_beyond_flag = keep_beyond_flag

        if self._vb:
            print('Minimum period:', self._sett_sel_phs_min_prd)
            print('Maximum period:', self._sett_sel_phs_max_prd)
            print('Keep beyond periods flag:', self._sett_sel_phs_beyond_flag)

            print_el()

        self._sett_sel_phs_set_flag = True
        return

    def set_initial_phase_spectra_settings(
            self,
            initial_phase_spectra_type,
            initial_phase_spectra):

        '''
        Specify initial phase spectra to use when phase annealing starts.

        Parameters
        ----------
        initial_phase_spectra_type : int
            The type of phase spectra supplied. 0 refers to a single spectra
            used as the initial one for all realizations. 1 refers to the case
            where each realtization gets a seperate initial spectra.
        initial_phase_spectra : list or tuple of 2D np.float64 np.ndarray
            A container holding the initial spectra. The shape of the
            spectra must correspond to that of the reference data. If N is the
            number of time steps in the reference data and M is the number of
            columns then each spectra should have the shape (N//2 + 1, M).
            The length of initial_phase_spectra should be 1 if
            initial_phase_spectra_type is 0 or equal to the number of
            realizations if initial_phase_spectra_type is 1. All values must
            lie inbetween -pi and +pi.
        '''

        if self._vb:
            print_sl()

            print('Setting initial phase spectra...\n')

        assert isinstance(initial_phase_spectra_type, int), (
            'initial_phase_spectra_type not an integer!')

        assert initial_phase_spectra_type in (0, 1), (
            'Invalid initial_phase_spectra_type!')

        assert isinstance(initial_phase_spectra, (tuple, list)), (
            'initial_phase_spectra not a list or a tuple!')

        assert len(initial_phase_spectra) > 0, (
            'Empty initial_phase_spectra!')

        for phs_spec in initial_phase_spectra:
            assert isinstance(phs_spec, np.ndarray), (
                'Phase spectra not a numpy array!')

            assert phs_spec.ndim == 2, 'Phase spectra not 2D!'

            assert phs_spec.dtype == np.float64, (
                'Incorrect data type of phase spectra!')

            assert np.all(np.isfinite(phs_spec)), (
                'Invalid values in phase spectra!')

            assert (np.all(phs_spec >= -np.pi) and
                    np.all(phs_spec <= +np.pi)), (
                        'Values in phase spectra out of range!')

        self._sett_init_phs_spec_type = initial_phase_spectra_type
        self._sett_init_phs_specs = tuple(initial_phase_spectra)

        if self._vb:
            print(
                'Initial phase spectra type:',
                self._sett_init_phs_spec_type)

            print_el()

        self._sett_init_phs_spec_set_flag = True
        return

    def set_limited_phase_randomization_settings(
            self,
            obj_val_lower_bound,
            obj_val_upper_bound,
            perturbation_lower_bound,
            perturbation_upper_bound,
            n_perturbation_intervals,
            iterations_per_attempt):

        '''
        Limit the amount of randomization applied to phases such that
        the resulting objective function values stay in given bounds.
        The idea being that with full spectrum phase randomization,
        the resulting objective value becomes so large that it cannot be
        brought down in a doable time period.

        This is done by perturbing all phases (excluding the phases
        that were set in set_mult_phase_settings) till the mean
        objective function value falls within given bounds by having smaller
        increments.

        Parameters
        ----------
        obj_val_lower_bound : float
            Lower bound of the objective function. Should be greater than
            zero and less than obj_val_upper_bound. After attempted limited
            phase randomization, the search stops if the mean objective
            value is between obj_val_lower_bound and obj_val_upper_bound.
        obj_val_upper_bound : float
            Upper bound of the objective function. Should be greater than
            obj_val_lower_bound and less than infinity!
        perturbation_lower_bound : float
            How much perturbation ratio to apply at the beginning. For
            example a generated phase is 0.25 Pi and perturbation_lower_bound
            is 0.01 than the resulting phase is shifted by 0.0025 Pi.
            Should be greater than zero.
        perturbation_upper_bound : float
            Upper bound of the perturbation to apply. Should be greater than
            perturbation_lower_bound and less than or equal to 1.0.
        n_perturbation_intervals : int
            The number of values to use in between perturbation_lower_bound
            and perturbation_upper_bound while finding the optimum amount
            of perturbation. Numpy linspace is used for this. Should be
            greater than zero.
        iterations_per_attempt : int
            Number of iterations to take the mean of per perturbation attempt.
            Should be greater than zero.
        '''

        if self._vb:
            print_sl()

            print('Setting limited phase randomization settings...\n')

        assert isinstance(obj_val_lower_bound, float), (
            'obj_val_lower_bound not a float!')

        assert isinstance(obj_val_upper_bound, float), (
            'obj_val_upper_bound not a float!')

        assert isinstance(perturbation_lower_bound, float), (
            'perturbation_lower_bound not a float!')

        assert isinstance(perturbation_upper_bound, float), (
            'perturbation_upper_bound not a float!')

        assert isinstance(n_perturbation_intervals, int), (
            'n_perturbation_intervals not an integer!')

        assert isinstance(iterations_per_attempt, int), (
            'iterations_per_attempt not an integer!')

        assert obj_val_lower_bound > 0, (
            'Invalid value of obj_val_lower_bound!')

        assert obj_val_lower_bound < np.inf, (
            'Invalid value of obj_val_lower_bound!')

        assert obj_val_upper_bound > obj_val_lower_bound, (
            'obj_val_upper_bound not greater than obj_val_lower_bound!')

        assert perturbation_lower_bound > 0, (
            'Invalid value of perturbation_lower_bound!')

        assert perturbation_upper_bound <= 1.0, (
            'Invalid value of perturbation_upper_bound!')

        assert perturbation_upper_bound > perturbation_lower_bound , (
            'perturbation_upper_bound not greater than '
            'perturbation_lower_bound!')

        assert n_perturbation_intervals > 0, (
            'Invalid value of n_perturbation_intervals!')

        assert iterations_per_attempt > 0, (
            'Invalid value of iterations_per_attempt!')

        self._sett_lim_phsrand_obj_lbd = obj_val_lower_bound
        self._sett_lim_phsrand_obj_ubd = obj_val_upper_bound
        self._sett_lim_phsrand_ptrb_lbd = perturbation_lower_bound
        self._sett_lim_phsrand_ptrb_ubd = perturbation_upper_bound
        self._sett_lim_phsrand_n_ptrb_vals = n_perturbation_intervals
        self._sett_lim_phsrand_iters_per_atpt = iterations_per_attempt

        if self._vb:
            print(
                'Objective function lower bound:',
                self._sett_lim_phsrand_obj_lbd)

            print(
                'Objective function upper bound:',
                self._sett_lim_phsrand_obj_ubd)

            print(
                'Perturbation lower bound:',
                self._sett_lim_phsrand_ptrb_lbd)

            print(
                'Perturbation upper bound:',
                self._sett_lim_phsrand_ptrb_ubd)

            print(
                'Total perturbation intervals:',
                self._sett_lim_phsrand_n_ptrb_vals)

            print(
                'Iterations per perturbation:',
                self._sett_lim_phsrand_iters_per_atpt)

            print_el()

        self._sett_lim_phsrand_set_flag = True
        return

    def verify(self):

        assert self._sett_ann_pa_sa_sett_flag, (
            'Call set_pa_misc_settings first!')

        GTGSettings._GTGSettings__verify(self)

        if self._sett_init_phs_spec_set_flag:

            if self._sett_init_phs_spec_type == 0:
                assert len(self._sett_init_phs_specs) == 1, (
                    'Phase spectra type and quantity do not match!')

            elif self._sett_init_phs_spec_type == 1:
                assert len(self._sett_init_phs_specs) == (
                    self._sett_misc_n_rltzns), (
                    'Phase spectra type and quantity do not match!')

            else:
                raise NotImplementedError

            for phs_spec in self._sett_init_phs_specs:
                assert phs_spec.shape[0] == (
                    1 + (self._data_ref_rltzn.shape[0] // 2)), (
                    'Shape of phase spectra not corresponding to that of the '
                    'reference data!')

        if self._sett_lim_phsrand_set_flag:
            self._sett_lim_phsrand_dir = (
                self._sett_misc_outs_dir / 'limited_perturbation')

        self._sett_ann_pa_sa_sett_verify_flag = True
        return

    __verify = verify
