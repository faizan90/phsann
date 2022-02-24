'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from gnrctsgenr import (
    GTGPrepareRltznRef,
    GTGPrepareRltznSim,
    GTGPrepareTfms,
    GTGPrepare,
    )


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

        if any([self._sett_obj_match_data_ft_flag,
                self._sett_obj_match_data_ms_ft_flag,
                self._sett_obj_match_data_ms_pair_ft_flag]):

            self._rs.data_ft_coeffs = np.fft.rfft(self._rs.data, axis=0)
            self._rs.data_ft_coeffs_mags = np.abs(self._rs.data_ft_coeffs)

            if self._sett_obj_match_data_ms_pair_ft_flag:
                self._rs.data_ft_coeffs_phss = np.angle(
                    self._rs.data_ft_coeffs)

        if any([self._sett_obj_match_probs_ft_flag,
                self._sett_obj_match_probs_ms_ft_flag,
                self._sett_obj_match_probs_ms_pair_ft_flag]):

            self._rs.probs_ft_coeffs = np.fft.rfft(self._rs.probs, axis=0)
            self._rs.probs_ft_coeffs_mags = np.abs(self._rs.probs_ft_coeffs)

            if self._sett_obj_match_probs_ms_pair_ft_flag:
                self._rs.probs_ft_coeffs_phss = np.angle(
                    self._rs.probs_ft_coeffs)

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
