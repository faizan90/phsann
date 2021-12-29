'''
Created on Dec 28, 2021

@author: Faizan3800X-Uni
'''

from time import asctime
from collections import deque
from timeit import default_timer

import numpy as np

from gnrctsgenr import (
    GTGBase,
    GTGData,
    GTGSettings,
    GTGPrepareRltznRef,
    GTGPrepareRltznSim,
    GTGPrepareBase,
    GTGPrepareTfms,
    GTGPrepareCDFS,
    GTGPrepareUpdate,
    GTGPrepare,
    GTGAlgBase,
    GTGAlgObjective,
    GTGAlgIO,
    GTGAlgLagNthWts,
    GTGAlgLabelWts,
    GTGAlgAutoObjWts,
    GTGAlgRealization,
    GTGAlgTemperature,
    GTGAlgMisc,
    GTGAlgorithm,
    GTGSave,
    )

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

        # Initial phase spectra.
        self._sett_init_phs_spec_type = None
        self._sett_init_phs_specs = None

        # Flags.
        self._sett_ann_pa_sa_sett_flag = False
        self._sett_mult_phs_flag = False
        self._sett_sel_phs_set_flag = False
        self._sett_init_phs_spec_set_flag = False
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

        self._sett_ann_pa_sa_sett_verify_flag = True
        return

    __verify = verify


class PhaseAnnealingPrepareRltznRef(GTGPrepareRltznRef):

    def __init__(self):

        GTGPrepareRltznRef.__init__(self)

        self.phs_sel_idxs = None
        self.phs_idxs = None
        return


class PhaseAnnealingPrepareRltznSim(GTGPrepareRltznSim):

    def __init__(self):

        GTGPrepareRltznSim.__init__(self)

        # To keep track of modified phases.
        self.phs_mod_flags = None

        self.phs_red_rates = None

        self.idxs_sclrs = None

        self.mag_spec_cdf = None
        return


class PhaseAnnealingPrepareTfms(GTGPrepareTfms):

    def __init__(self):

        GTGPrepareTfms.__init__(self)
        return

    def _get_sim_ft_pln(self):

        '''
        Plain phase randomization.
        '''

        ft = np.zeros(self._rs.shape, dtype=np.complex)

        mag_spec = self._rr.mag_spec.copy()

        if self._sett_init_phs_spec_set_flag:

            if ((self._sett_init_phs_spec_type == 0) or
                (self._alg_rltzn_iter is None)):

                # Outside of annealing or when only one initial phs_spec.
                new_phss = self._sett_init_phs_specs[0]

            elif self._sett_init_phs_spec_type == 1:
                new_phss = self._sett_init_phs_specs[self._alg_rltzn_iter]

            else:
                raise NotImplementedError

            phs_spec = new_phss[1:-1,:].copy()

        else:
            rands = np.random.random((self._rs.shape[0] - 2, 1))

            rands = 1.0 * (-np.pi + (2 * np.pi * rands))

            rands[~self._rr.phs_sel_idxs] = 0.0

            phs_spec = self._rr.phs_spec[1:-1,:].copy()

            phs_spec += rands  # out of bound phs

        ft.real[1:-1,:] = mag_spec[1:-1,:] * np.cos(phs_spec)
        ft.imag[1:-1,:] = mag_spec[1:-1,:] * np.sin(phs_spec)

        self._rs.phs_mod_flags[1:-1,:] += 1

        # First and last coefficients are not written to anywhere, normally.
        ft[+0] = self._rr.ft[+0].copy()
        ft[-1] = self._rr.ft[-1].copy()

        return ft


class PhaseAnnealingPrepare(GTGPrepare):

    def __init__(self):

        GTGPrepare.__init__(self)
        return

    def _set_sel_phs_idxs(self):

        periods = self._rr.probs.shape[0] / (
            np.arange(1, self._rr.ft.shape[0] - 1))

        self._rr.phs_sel_idxs = np.ones(
            self._rr.ft.shape[0] - 2, dtype=bool)

        if self._sett_sel_phs_min_prd is not None:
            assert periods.min() <= self._sett_sel_phs_min_prd, (
                'Minimum period does not exist in data!')

            assert periods.max() > self._sett_sel_phs_min_prd, (
                'Data maximum period greater than or equal to min_period!')

            self._rr.phs_sel_idxs[
                periods < self._sett_sel_phs_min_prd] = False

        if self._sett_sel_phs_max_prd is not None:
            assert periods.max() >= self._sett_sel_phs_max_prd, (
                'Maximum period does not exist in data!')

            self._rr.phs_sel_idxs[
                periods > self._sett_sel_phs_max_prd] = False

        assert self._rr.phs_sel_idxs.sum(), (
            'Incorrect min_period or max_period, '
            'not phases selected for phsann!')
        return

    def _gen_ref_aux_data(self):

        self._gen_ref_aux_data_gnrc()

        self._set_sel_phs_idxs()

        self._rr.phs_idxs = np.arange(
            1, self._rr.ft.shape[0] - 1)[self._rr.phs_sel_idxs]

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._rs.shape = (1 + (self._data_ref_shape[0] // 2),
            self._data_ref_n_labels)

        self._rs.n_idxs_all_cts = np.zeros(
            self._rs.shape[0], dtype=np.uint64)

        self._rs.n_idxs_acpt_cts = np.zeros(
            self._rs.shape[0], dtype=np.uint64)

        # Has to be before the call to _get_sim_ft_pln.
        if self._rs.phs_mod_flags is None:
            self._rs.phs_mod_flags = np.zeros(self._rs.shape, dtype=int)

            self._rs.phs_mod_flags[+0,:] += 1
            self._rs.phs_mod_flags[-1,:] += 1

        ft = self._get_sim_ft_pln()

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'

        data = np.fft.irfft(ft, axis=0)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        self._rs.ft = ft
        self._rs.phs_spec = np.angle(ft)
        self._rs.mag_spec = np.abs(ft)

        self._update_obj_vars('sim')

        if self._sett_ann_mag_spec_cdf_idxs_flag:
            mag_spec = self._rs.mag_spec.copy()

            mag_spec = mag_spec[self._rr.phs_idxs]

            mag_spec_pdf = mag_spec.sum(axis=1) / mag_spec.sum()

            assert np.all(mag_spec_pdf > 0), (
                'Phases with zero magnitude not allowed!')

            mag_spec_pdf = 1 / mag_spec_pdf

            mag_spec_pdf /= mag_spec_pdf.sum()

            assert np.all(np.isfinite(mag_spec_pdf)), (
                'Invalid values in mag_spec_pdf!')

            mag_spec_cdf = mag_spec_pdf.copy()

            self._rs.mag_spec_cdf = mag_spec_cdf

        self._prep_sim_aux_flag = True
        return

    def verify(self):

        GTGPrepare._GTGPrepare__verify(self)
        return

    __verify = verify


class PhaseAnnealingAlgLagNthWts(GTGAlgLagNthWts):

    def __init__(self):

        GTGAlgLagNthWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_lag_nth_wts(self, phs_red_rate, idxs_sclr):

        self._init_lag_nth_wts()

        self._alg_wts_lag_nth_search_flag = True

        for _ in range(self._sett_wts_lags_nths_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_lag_nth_search_flag = False

        self._update_lag_nth_wts()
        return


class PhaseAnnealingAlgLabelWts(GTGAlgLabelWts):

    def __init__(self):

        GTGAlgLabelWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_label_wts(self, phs_red_rate, idxs_sclr):

        self._init_label_wts()

        self._alg_wts_label_search_flag = True

        for _ in range(self._sett_wts_label_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_label_search_flag = False

        self._update_label_wts()
        return


class PhaseAnnealingAlgAutoObjWts(GTGAlgAutoObjWts):

    def __init__(self):

        GTGAlgAutoObjWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_auto_obj_wts(self, phs_red_rate, idxs_sclr):

        self._sett_wts_obj_wts = None
        self._alg_wts_obj_raw = []
        self._alg_wts_obj_search_flag = True

        for _ in range(self._sett_wts_obj_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_obj_raw = np.array(
            self._alg_wts_obj_raw, dtype=np.float64)

        assert self._alg_wts_obj_raw.ndim == 2
        assert self._alg_wts_obj_raw.shape[0] > 1

        self._update_obj_wts()

        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False
        return


class PhaseAnnealingRealization(GTGAlgRealization):

    def __init__(self):

        GTGAlgRealization.__init__(self)
        return

    def _update_wts(self):

        self._update_wts_phsann(1.0, 1.0)
        return

    def _update_wts_phsann(self, phs_red_rate, idxs_sclr):

        if self._sett_wts_lags_nths_set_flag:

            if self._vb:
                print_sl()

                print(f'Computing lag and nths weights...')

            self._set_lag_nth_wts(phs_red_rate, idxs_sclr)

            if self._vb:

                self._show_lag_nth_wts()

                print(f'Done computing lag and nths weights.')

                print_el()

        if self._sett_wts_label_set_flag:

            if self._vb:
                print_sl()

                print(f'Computing label weights...')

            self._set_label_wts(phs_red_rate, idxs_sclr)

            if self._vb:
                self._show_label_wts()

                print(f'Done computing label weights.')

                print_el()

        if self._sett_wts_obj_auto_set_flag:
            if self._vb:
                print_sl()

                print(f'Computing individual objective function weights...')

            self._set_auto_obj_wts(phs_red_rate, idxs_sclr)

            if self._vb:
                self._show_obj_wts()

                print(f'Done computing individual objective function weights.')

                print_el()

        return

    def _show_rltzn_situ(
            self,
            iter_ctr,
            rltzn_iter,
            iters_wo_acpt,
            tol,
            temp,
            phs_red_rate,
            acpt_rate,
            new_obj_val,
            obj_val_min,
            iter_wo_min_updt):

        c1 = self._sett_ann_max_iters >= 10000
        c2 = not (iter_ctr % (0.1 * self._sett_ann_max_iters))

        if (c1 and c2) or (iter_ctr == 1):
            with self._lock:
                print_sl()

                print(
                    f'Realization {rltzn_iter} finished {iter_ctr} out of '
                    f'{self._sett_ann_max_iters} iterations on {asctime()}.')

                print(f'Current objective function value: {new_obj_val:9.2E}')

                print(
                    f'Running minimum objective function value: '
                    f'{obj_val_min:9.2E}\n')

                iter_wo_min_updt_ratio = (
                    iter_wo_min_updt / self._sett_ann_max_iter_wo_min_updt)

                print(
                    f'Stopping criteria variables:\n'
                    f'{self._alg_cnsts_stp_crit_labs[0]}: '
                    f'{iter_ctr/self._sett_ann_max_iters:6.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[1]}: '
                    f'{iters_wo_acpt/self._sett_ann_max_iter_wo_chng:6.2%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[2]}: {tol:9.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[3]}: {temp:9.2E}\n'
                    f'{self._alg_cnsts_stp_crit_labs[4]}: '
                    f'{phs_red_rate:6.3%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[5]}: {acpt_rate:6.3%}\n'
                    f'{self._alg_cnsts_stp_crit_labs[6]}: '
                    f'{iter_wo_min_updt_ratio:6.2%}')

                print_el()
        return

    def _get_stopp_criteria(self, test_vars):

        (iter_ctr,
         iters_wo_acpt,
         tol,
         temp,
         phs_red_rate,
         acpt_rate,
         iter_wo_min_updt) = test_vars

        stopp_criteria = (
            (iter_ctr < self._sett_ann_max_iters),
            (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol),
            (temp > self._alg_cnsts_almost_zero),
            (phs_red_rate > self._sett_ann_min_phs_red_rate),
            (acpt_rate > self._sett_ann_stop_acpt_rate),
            (iter_wo_min_updt < self._sett_ann_max_iter_wo_min_updt),
            )

        if iter_ctr <= 1:
            assert len(self._alg_cnsts_stp_crit_labs) == len(stopp_criteria), (
                'stopp_criteria and its labels are not of the '
                'same length!')

        return stopp_criteria

    def _get_phs_red_rate(self, iter_ctr, acpt_rate, old_phs_red_rate):

        _ = old_phs_red_rate  # To avoid the annoying unused warning.

        if self._alg_ann_runn_auto_init_temp_search_flag:
            phs_red_rate = 1.0

        else:
            if self._sett_ann_phs_red_rate_type == 0:
                phs_red_rate = 1.0

            elif self._sett_ann_phs_red_rate_type == 1:
                phs_red_rate = 1.0 - (iter_ctr / self._sett_ann_max_iters)

            elif self._sett_ann_phs_red_rate_type == 2:
                phs_red_rate = float((
                    self._sett_ann_phs_red_rate **
                    (iter_ctr // self._sett_ann_upt_evry_iter)))

            elif self._sett_ann_phs_red_rate_type == 3:

                # An unstable mean of acpts_rjts_dfrntl is a
                # problem. So, it has to be long enough.

                # Why the min(acpt_rate, old_phs_red_rate) was used?

                # Not using min might result in instability as acpt_rate will
                # oscillate when phs_red_rate oscillates but this is taken
                # care of by the maximum iterations without updating the
                # global minimum.

                # Also, it might get stuck in a local minimum by taking min.

                # Normally, it moves very slowly if min is used after some
                # iterations. The accpt_rate stays high due to this slow
                # movement after hitting a low. This becomes a substantial
                # part of the time taken to finish annealing which doesn't
                # bring much improvement to the global minimum.

                # phs_red_rate = max(
                #     self._sett_ann_min_phs_red_rate,
                #     min(acpt_rate, old_phs_red_rate))

                phs_red_rate = acpt_rate

            else:
                raise NotImplemented(
                    'Unknown _sett_ann_phs_red_rate_type:',
                    self._sett_ann_phs_red_rate_type)

            assert phs_red_rate >= 0.0, 'Invalid phs_red_rate!'

        return phs_red_rate

    def _get_phs_idxs_sclr(self, iter_ctr, acpt_rate, old_idxs_sclr):

        _ = old_idxs_sclr  # To avoid the annoying unused warning.

        if not self._sett_mult_phs_flag:
            idxs_sclr = np.nan

        else:
            if self._sett_mult_phs_sample_type == 0:
                idxs_sclr = 1.0

            elif self._sett_mult_phs_sample_type == 1:
                idxs_sclr = 1.0 - (iter_ctr / self._sett_ann_max_iters)

            elif self._sett_mult_phs_sample_type == 2:
                idxs_sclr = float((
                    self._sett_mult_phss_red_rate **
                    (iter_ctr // self._sett_ann_upt_evry_iter)))

            elif self._sett_mult_phs_sample_type == 3:
                # Same story as that of _get_phs_red_rate.
                idxs_sclr = acpt_rate

            else:
                raise NotImplementedError

            assert np.isfinite(idxs_sclr), f'Invalid idxs_sclr ({idxs_sclr})!'

        return idxs_sclr

    def _get_next_idxs(self, idxs_sclr):

        # _sim_mag_spec_cdf makes it difficult without a while-loop.

        idxs_diff = self._rr.phs_sel_idxs.sum()

        assert idxs_diff > 0, idxs_diff

        if any([
            self._alg_wts_lag_nth_search_flag,
            self._alg_wts_label_search_flag,
            self._alg_wts_obj_search_flag,
            self._alg_ann_runn_auto_init_temp_search_flag,
            ]):

            # Full spectrum randomization during search.
            new_idxs = np.arange(1, self._rs.shape[0] - 1)

        else:
            if self._sett_mult_phs_flag:
                min_idx_to_gen = self._sett_mult_phs_n_beg_phss
                max_idxs_to_gen = self._sett_mult_phs_n_end_phss

            else:
                min_idx_to_gen = 1
                max_idxs_to_gen = 2

            # Inclusive.
            min_idxs_to_gen = min([min_idx_to_gen, idxs_diff])

            # Inclusive.
            max_idxs_to_gen = min([max_idxs_to_gen, idxs_diff])

            if np.isnan(idxs_sclr):
                idxs_to_gen = np.random.randint(
                    min_idxs_to_gen, max_idxs_to_gen)

            else:
                idxs_to_gen = min_idxs_to_gen + (
                    int(round(idxs_sclr *
                        (max_idxs_to_gen - min_idxs_to_gen))))

            assert min_idx_to_gen >= 1, 'This shouldn\'t have happend!'
            assert idxs_to_gen >= 1, 'This shouldn\'t have happend!'

            if min_idx_to_gen == idxs_diff:
                new_idxs = np.arange(1, min_idxs_to_gen + 1)

            else:
                new_idxs = []
                sample = self._rr.phs_idxs

                if self._sett_ann_mag_spec_cdf_idxs_flag:
                    new_idxs = np.random.choice(
                        sample,
                        idxs_to_gen,
                        replace=False,
                        p=self._rs.mag_spec_cdf)

                else:
                    new_idxs = np.random.choice(
                        sample,
                        idxs_to_gen,
                        replace=False)

        assert np.all(0 < new_idxs)
        assert np.all(new_idxs < (self._rs.shape[0] - 1))

        return new_idxs

    @GTGBase._timer_wrap
    def _get_next_iter_vars(self, phs_red_rate, idxs_sclr):

        new_idxs = self._get_next_idxs(idxs_sclr)

        if True:
            # Phase Annealing.

            # Making a copy of the phases is important if not then the
            # returned old_phs and new_phs are SOMEHOW the same.
            old_phss = self._rs.phs_spec[new_idxs,:].copy()

            new_phss = -np.pi + (
                2 * np.pi * np.random.random((old_phss.shape[0], 1)))

            if self._alg_ann_runn_auto_init_temp_search_flag:
                pass

            else:
                new_phss *= phs_red_rate

            new_phss = old_phss + new_phss

            new_rect_phss = np.full(new_phss.shape, np.nan)

            for i in range(new_phss.shape[0]):
                for j in range(new_phss.shape[1]):

                    # Didn't work without copy.
                    new_phs = new_phss[i, j].copy()

                    if new_phs > +np.pi:
                        ratio = (new_phs / +np.pi) - 1
                        new_phs = -np.pi * (1 - ratio)

                    elif new_phs < -np.pi:
                        ratio = (new_phs / -np.pi) - 1
                        new_phs = +np.pi * (1 - ratio)

                    assert (-np.pi <= new_phs <= +np.pi)

                    new_rect_phss[i, j] = new_phs

            assert np.all(np.isfinite(new_rect_phss)), 'Invalid phases!'

            new_phss = new_rect_phss

            old_coeffs = new_coeffs = None

        else:  # Magnnealing.
            assert not self._sett_init_phs_spec_set_flag, (
                'Not implemented for initial phase spectra!')

            old_phss = new_phss = None

            old_coeffs = self._rs.ft[new_idxs,:].copy()

            mags = np.abs(old_coeffs) + (
                (-1 + 2 * np.random.random(old_coeffs.shape)) * phs_red_rate)

            le_zero_idxs = mags < 0

            mags[le_zero_idxs] = -mags[le_zero_idxs]

            phss = np.angle(old_coeffs)

            new_coeffs = np.full_like(old_coeffs, np.nan)
            new_coeffs.real = mags * np.cos(phss)
            new_coeffs.imag = mags * np.sin(phss)

        return old_phss, new_phss, old_coeffs, new_coeffs, new_idxs

    def _update_sim_no_prms(self):

        data = np.fft.irfft(self._rs.ft, axis=0)

        probs = self._get_probs(data, True)

        self._rs.data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._rs.data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._rs.probs = probs

        self._update_obj_vars('sim')
        return

    @GTGBase._timer_wrap
    def _update_sim(self, idxs, phss, coeffs, load_snapshot_flag):

        if coeffs is not None:
            self._rs.ft[idxs] = coeffs
            self._rs.mag_spec[idxs] = np.abs(self._rs.ft[idxs,:])

        else:
            self._rs.phs_spec[idxs] = phss

            self._rs.ft.real[idxs] = np.cos(phss) * self._rs.mag_spec[idxs]
            self._rs.ft.imag[idxs] = np.sin(phss) * self._rs.mag_spec[idxs]

        if load_snapshot_flag:
            self._load_snapshot()

        else:
            self._update_sim_no_prms()

        return

    def _gen_gnrc_rltzn(self, args):

        (rltzn_iter,
         init_temp,
        ) = args

        assert self._alg_verify_flag, 'Call verify first!'

        beg_time = default_timer()

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            temp = init_temp

        else:
            # _alg_rltzn_iter should be only set when annealing is started.
            self._alg_rltzn_iter = rltzn_iter

            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                    'Invalid rltzn_iter!')

            temp = self._sett_ann_init_temp

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implemention for 2D only!')

        # Randomize all phases before starting.
        self._gen_sim_aux_data()

        # Initialize sim anneal variables.
        iter_ctr = 0

        if self._sett_ann_auto_init_temp_trgt_acpt_rate is not None:
            acpt_rate = self._sett_ann_auto_init_temp_trgt_acpt_rate

        else:
            acpt_rate = 1.0

        # Phase Annealing variable.
        phs_red_rate = self._get_phs_red_rate(iter_ctr, acpt_rate, 1.0)

        # Phase Annealing variable.
        idxs_sclr = self._get_phs_idxs_sclr(iter_ctr, acpt_rate, 1.0)

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            iters_wo_acpt = 0
            tol = np.inf

            iter_wo_min_updt = 0

            tols_dfrntl = deque(maxlen=self._sett_ann_obj_tol_iters)

            acpts_rjts_dfrntl = deque(maxlen=self._sett_ann_acpt_rate_iters)

            stopp_criteria = self._get_stopp_criteria(
                (iter_ctr,
                 iters_wo_acpt,
                 tol,
                 temp,
                 phs_red_rate,  # Phase Annealing variable.
                 acpt_rate,
                 iter_wo_min_updt))

        old_idxs = self._get_next_idxs(idxs_sclr)
        new_idxs = old_idxs

        old_obj_val = self._get_obj_ftn_val().sum()

        self._update_snapshot()

        # Initialize diagnostic variables.
        acpts_rjts_all = []

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            tols = []

            obj_vals_all = []

            obj_val_min = old_obj_val
            obj_vals_min = []

            # Phase Annealing variable.
            phs_red_rates = [[iter_ctr, phs_red_rate]]

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

            # Phase Annealing variable.
            idxs_sclrs = [[iter_ctr, idxs_sclr]]

            obj_vals_all_indiv = []

        else:
            pass

        self._rs.ft_best = self._rs.ft.copy()

        while all(stopp_criteria):

            #==============================================================
            # Simulated annealing start.
            #==============================================================

            # Phase Annealing variables.
            (old_phss,
             new_phss,
             old_coeffs,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            new_obj_val_indiv = self._get_obj_ftn_val()
            new_obj_val = new_obj_val_indiv.sum()

            old_new_diff = old_obj_val - new_obj_val

            if old_new_diff > 0:
                accept_flag = True

            else:
                rand_p = np.random.random()

                boltz_p = np.exp(old_new_diff / temp)

                if rand_p < boltz_p:
                    accept_flag = True

                else:
                    accept_flag = False

            if self._alg_force_acpt_flag:
                accept_flag = True
                self._alg_force_acpt_flag = False

            if accept_flag:
                old_idxs = new_idxs

                old_obj_val = new_obj_val

                self._update_snapshot()

            else:
                self._update_sim(new_idxs, old_phss, old_coeffs, True)

            iter_ctr += 1

            #==============================================================
            # Simulated annealing end.
            #==============================================================

            acpts_rjts_all.append(accept_flag)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                stopp_criteria = (
                    (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                    )

            else:
                obj_vals_all_indiv.append(new_obj_val_indiv)

                if new_obj_val < obj_val_min:
                    iter_wo_min_updt = 0

                    self._rs.ft_best = self._rs.ft.copy()

                else:
                    iter_wo_min_updt += 1

                tols_dfrntl.append(abs(old_new_diff))

                obj_val_min = min(obj_val_min, new_obj_val)

                obj_vals_min.append(obj_val_min)
                obj_vals_all.append(new_obj_val)

                acpts_rjts_dfrntl.append(accept_flag)

                self._rs.n_idxs_all_cts[new_idxs] += 1

                if iter_ctr >= tols_dfrntl.maxlen:
                    tol = sum(tols_dfrntl) / float(tols_dfrntl.maxlen)

                    assert np.isfinite(tol), 'Invalid tol!'

                    tols.append(tol)

                if accept_flag:
                    self._rs.n_idxs_acpt_cts[new_idxs] += 1
                    iters_wo_acpt = 0

                else:
                    iters_wo_acpt += 1

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    acpt_rates_dfrntl.append([iter_ctr - 1, acpt_rate])

                    acpt_rate = (
                        sum(acpts_rjts_dfrntl) /
                        float(acpts_rjts_dfrntl.maxlen))

                    acpt_rates_dfrntl.append([iter_ctr, acpt_rate])

                if (iter_ctr % self._sett_ann_upt_evry_iter) == 0:

                    # Temperature.
                    temps.append([iter_ctr - 1, temp])

                    temp *= self._sett_ann_temp_red_rate

                    assert temp >= 0.0, 'Invalid temp!'

                    temps.append([iter_ctr, temp])

                    # Phase Annealing variable.
                    # Phase reduction rate.
                    phs_red_rates.append([iter_ctr - 1, phs_red_rate])

                    phs_red_rate = self._get_phs_red_rate(
                        iter_ctr, acpt_rate, phs_red_rate)

                    phs_red_rates.append([iter_ctr, phs_red_rate])

                    # Phase Annealing variable.
                    # Phase indices reduction rate.
                    idxs_sclrs.append([iter_ctr - 1, idxs_sclr])

                    idxs_sclr = self._get_phs_idxs_sclr(
                        iter_ctr, acpt_rate, idxs_sclr)

                    idxs_sclrs.append([iter_ctr, idxs_sclr])

                if self._vb:
                    self._show_rltzn_situ(
                        iter_ctr,
                        rltzn_iter,
                        iters_wo_acpt,
                        tol,
                        temp,
                        phs_red_rate,  # Phase Annealing variable.
                        acpt_rate,
                        new_obj_val,
                        obj_val_min,
                        iter_wo_min_updt)

                stopp_criteria = self._get_stopp_criteria(
                    (iter_ctr,
                     iters_wo_acpt,
                     tol,
                     temp,
                     phs_red_rate,  # Phase Annealing variable.
                     acpt_rate,
                     iter_wo_min_updt))

        # Manual update of timer because this function writes timings
        # to the HDF5 file before it returns.
        if '_gen_gnrc_rltzn' not in self._dur_tmr_cumm_call_times:
            self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] = 0.0
            self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] = 0.0

        self._dur_tmr_cumm_call_times['_gen_gnrc_rltzn'] += (
            default_timer() - beg_time)

        self._dur_tmr_cumm_n_calls['_gen_gnrc_rltzn'] += 1

        if self._alg_ann_runn_auto_init_temp_search_flag:

            ret = sum(acpts_rjts_all) / len(acpts_rjts_all), temp

        else:
            assert self._rs.n_idxs_all_cts[+0] == 0
            assert self._rs.n_idxs_all_cts[-1] == 0

            # _sim_ft set to _sim_ft_best in _update_sim_at_end.
            self._update_ref_at_end()
            self._update_sim_at_end()

            self._rs.label = (
                f'{rltzn_iter:0{len(str(self._sett_misc_n_rltzns))}d}')

            self._rs.iter_ctr = iter_ctr
            self._rs.iters_wo_acpt = iters_wo_acpt
            self._rs.tol = tol
            self._rs.temp = temp
            self._rs.stopp_criteria = np.array(stopp_criteria)
            self._rs.tols = np.array(tols, dtype=np.float64)
            self._rs.obj_vals_all = np.array(obj_vals_all, dtype=np.float64)

            self._rs.acpts_rjts_all = np.array(acpts_rjts_all, dtype=bool)

            self._rs.acpt_rates_all = (
                np.cumsum(self._rs.acpts_rjts_all) /
                np.arange(1, self._rs.acpts_rjts_all.size + 1, dtype=float))

            self._rs.obj_vals_min = np.array(obj_vals_min, dtype=np.float64)

            self._rs.temps = np.array(temps, dtype=np.float64)

            # Phase Annealing variable.
            self._rs.phs_red_rates = np.array(phs_red_rates, dtype=np.float64)

            self._rs.acpt_rates_dfrntl = np.array(
                acpt_rates_dfrntl, dtype=np.float64)

            self._rs.ref_sim_ft_corr = self._get_cumm_ft_corr(
                self._rr.ft, self._rs.ft).astype(np.float64)

            self._rs.sim_sim_ft_corr = self._get_cumm_ft_corr(
                    self._rs.ft, self._rs.ft).astype(np.float64)

            self._rs.obj_vals_all_indiv = np.array(
                obj_vals_all_indiv, dtype=np.float64)

            # Phase Annealing variable.
            self._rs.idxs_sclrs = np.array(idxs_sclrs, dtype=np.float64)

            self._rs.cumm_call_durations = self._dur_tmr_cumm_call_times
            self._rs.cumm_n_calls = self._dur_tmr_cumm_n_calls

            # Phase Annealing variable.
            assert np.all(self._rs.phs_mod_flags >= 1), (
                'Some phases were not modified!')

            self._write_cls_rltzn()

            ret = stopp_criteria

        self._alg_snapshot = None
        return ret


class PhaseAnnealingMain(
        GTGBase,
        GTGData,
        PhaseAnnealingSettings,
        GTGPrepareBase,
        PhaseAnnealingPrepareTfms,
        GTGPrepareCDFS,
        GTGPrepareUpdate,
        PhaseAnnealingPrepare,
        GTGAlgBase,
        GTGAlgObjective,
        GTGAlgIO,
        PhaseAnnealingAlgLagNthWts,
        PhaseAnnealingAlgLabelWts,
        PhaseAnnealingAlgAutoObjWts,
        PhaseAnnealingRealization,
        GTGAlgTemperature,
        GTGAlgMisc,
        GTGAlgorithm,
        GTGSave):

    def __init__(self, verbose):

        GTGBase.__init__(self, verbose)
        GTGData.__init__(self)
        PhaseAnnealingSettings.__init__(self)

        self._rr = PhaseAnnealingPrepareRltznRef()  # Reference.
        self._rs = PhaseAnnealingPrepareRltznSim()  # Simulation.

        GTGPrepareBase.__init__(self)
        PhaseAnnealingPrepareTfms
        GTGPrepareCDFS
        GTGPrepareUpdate
        PhaseAnnealingPrepare
        GTGAlgBase.__init__(self)
        GTGAlgObjective.__init__(self)
        GTGAlgIO.__init__(self)
        PhaseAnnealingAlgLagNthWts.__init__(self)
        PhaseAnnealingAlgLabelWts.__init__(self)
        PhaseAnnealingAlgAutoObjWts.__init__(self)
        PhaseAnnealingRealization.__init__(self)
        GTGAlgTemperature.__init__(self)
        GTGAlgMisc.__init__(self)
        GTGAlgorithm.__init__(self)
        GTGSave.__init__(self)

        self._main_verify_flag = False
        return

    def verify(self):

        GTGData._GTGData__verify(self)

        PhaseAnnealingSettings._PhaseAnnealingSettings__verify(self)

        assert self._sett_ann_pa_sa_sett_verify_flag, (
            'Phase Aneealing settings in an unverfied state!')

        PhaseAnnealingPrepare._PhaseAnnealingPrepare__verify(self)
        GTGAlgorithm._GTGAlgorithm__verify(self)
        GTGSave._GTGSave__verify(self)

        assert self._save_verify_flag, 'Save in an unverified state!'

        self._main_verify_flag = True
        return
