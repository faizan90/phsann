'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np
from scipy.stats import norm

from .updt import PhaseAnnealingPrepareUpdate as PAPU
from ...misc import (
    print_sl,
    print_el,
    )


class PhaseAnnealingPrepare(PAPU):

    '''Prepare derived variables required by phase annealing here.'''

    def __init__(self, verbose=True):

        PAPU.__init__(self, verbose)
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

    def _get_data_tfm(self, data, probs):

        assert self._data_tfm_type in self._data_tfm_types, (
            f'Unknown data transform string {self._data_tfm_type}!')

        if self._data_tfm_type == 'log_data':
            data_tfm = np.log(data)

        elif self._data_tfm_type == 'probs':
            data_tfm = probs.copy()

        elif self._data_tfm_type == 'data':
            data_tfm = data.copy()

        elif self._data_tfm_type == 'probs_sqrt':
            data_tfm = probs ** 0.5

        elif self._data_tfm_type == 'norm':
            data_tfm = norm.ppf(probs)

        else:
            raise NotImplementedError(
                f'_data_tfm_type can only be from: '
                f'{self._data_tfm_types}!')

        assert np.all(np.isfinite(data_tfm)), 'Invalid values in data_tfm!'

        return data_tfm

    def _gen_ref_aux_data(self):

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        probs = self._get_probs(self._data_ref_rltzn, False)

        self._rr.data_tfm = self._get_data_tfm(self._data_ref_rltzn, probs)

        ft = np.fft.rfft(self._rr.data_tfm, axis=0)

        self._rr.data = self._data_ref_rltzn.copy()

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'
        assert np.all(np.isfinite(mag_spec)), 'Invalid values in mag_spec!'

        self._rr.probs = probs
        self._rr.probs_srtd = np.sort(probs, axis=0)

        self._rr.ft = ft
        self._rr.phs_spec = phs_spec
        self._rr.mag_spec = mag_spec

        if self._sett_obj_cos_sin_dist_flag:
            self._rr.cos_sin_cdfs_dict = self._get_cos_sin_cdfs_dict(
                self._rr.ft)

        self._update_obj_vars('ref')

        self._set_sel_phs_idxs()

        self._rr.phs_idxs = np.arange(
            1, self._rr.ft.shape[0] - 1)[self._rr.phs_sel_idxs]

        self._rr.ft_cumm_corr = self._get_cumm_ft_corr(
            self._rr.ft, self._rr.ft)

        if self._sett_obj_use_obj_dist_flag:
            if self._sett_obj_scorr_flag:
                self._rr.scorr_diffs_cdfs_dict = (
                    self._get_scorr_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_1_flag:
                self._rr.asymm_1_diffs_cdfs_dict = (
                    self._get_asymm_1_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_asymm_type_2_flag:
                self._rr.asymm_2_diffs_cdfs_dict = (
                    self._get_asymm_2_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_ecop_dens_flag:
                self._rr.ecop_dens_diffs_cdfs_dict = (
                    self._get_ecop_dens_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_ecop_etpy_flag:
                self._rr.ecop_etpy_diffs_cdfs_dict = (
                    self._get_ecop_etpy_diffs_cdfs_dict(self._rr.probs))

            if self._sett_obj_nth_ord_diffs_flag:
                self._rr.nth_ord_diffs_cdfs_dict = (
                    self._get_nth_ord_diffs_cdfs_dict(
                        self._rr.data, self._sett_obj_nth_ords_vld))

            if self._sett_obj_pcorr_flag:
                self._rr.pcorr_diffs_cdfs_dict = (
                    self._get_pcorr_diffs_cdfs_dict(self._rr.data))

        if self._sett_obj_asymm_type_1_ft_flag:
            self._rr.asymm_1_diffs_ft_dict = (
                self._get_asymm_1_diffs_ft_dict(self._rr.probs))

        if self._sett_obj_asymm_type_2_ft_flag:
            self._rr.asymm_2_diffs_ft_dict = (
                self._get_asymm_2_diffs_ft_dict(self._rr.probs))

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._rr.nth_ord_diffs_ft_dict = (
                self._get_nth_ord_diffs_ft_dict(
                    self._rr.data, self._sett_obj_nth_ords_vld))

        if self._sett_obj_etpy_ft_flag:
            self._rr.etpy_ft_dict = (
                self._get_etpy_ft_dict(self._rr.probs))

        if self._data_ref_n_labels > 1:
            # NOTE: don't add flags here
            self._rr.mult_asymm_1_diffs_cdfs_dict = (
                self._get_mult_asymm_1_diffs_cdfs_dict(self._rr.probs))

            self._rr.mult_asymm_2_diffs_cdfs_dict = (
                self._get_mult_asymm_2_diffs_cdfs_dict(self._rr.probs))

            self._rr.mult_ecop_dens_cdfs_dict = (
                self._get_mult_ecop_dens_diffs_cdfs_dict(self._rr.probs))

            self._rr.mult_asymm_1_cmpos_ft_dict = (
                self._get_mult_asymm_1_cmpos_ft(self._rr.probs, 'ref'))

            self._rr.mult_asymm_2_cmpos_ft_dict = (
                self._get_mult_asymm_2_cmpos_ft(self._rr.probs, 'ref'))

            self._rr.mult_etpy_cmpos_ft_dict = (
                self._get_mult_etpy_cmpos_ft(self._rr.probs, 'ref'))

#             self._get_mult_scorr_cmpos_ft(self._rr.probs, 'ref')

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._rs.shape = (1 + (self._data_ref_shape[0] // 2),
            self._data_ref_n_labels)

        if self._rs.phs_mod_flags is None:
            self._rs.phs_mod_flags = np.zeros(self._rs.shape, dtype=int)

            self._rs.phs_mod_flags[+0,:] += 1
            self._rs.phs_mod_flags[-1,:] += 1

            self._rs.n_idxs_all_cts = np.zeros(
                self._rs.shape[0], dtype=np.uint64)

            self._rs.n_idxs_acpt_cts = np.zeros(
                self._rs.shape[0], dtype=np.uint64)

        ft, mag_spec_flags = self._get_sim_ft_pln()

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

        self._rs.mag_spec_flags = mag_spec_flags

        self._update_obj_vars('sim')

        self._rs.mag_spec_idxs = np.argsort(
            self._rs.mag_spec[1:], axis=0)[::-1,:]

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

    def prepare(self):

        '''Generate data required before phase annealing starts'''

        PAPU._PhaseAnnealingSettings__verify(self)
        assert self._sett_verify_flag, 'Settings in an unverfied state!'

        self._gen_ref_aux_data()
        assert self._prep_ref_aux_flag, (
            'Apparently, _gen_ref_aux_data did not finish as expected!')

        self._gen_sim_aux_data()
        assert self._prep_sim_aux_flag, (
            'Apparently, _gen_sim_aux_data did not finish as expected!')

        self._prep_prep_flag = True
        return

    def verify(self):

        assert self._prep_prep_flag, 'Call prepare first!'

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify
