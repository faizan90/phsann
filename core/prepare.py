'''
Created on Dec 27, 2019

@author: Faizan
'''
from math import ceil as mceil
from collections import namedtuple

import numpy as np
from scipy.stats import rankdata, norm, expon
from scipy.interpolate import interp1d

from ..misc import print_sl, print_el, roll_real_2arrs
from ..cyth import (
    get_asymms_sample,
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    )

from .settings import PhaseAnnealingSettings as PAS


class PhaseAnnealingPrepare(PAS):

    '''Prepare derived variables required by phase annealing here'''

    def __init__(self, verbose=True):

        PAS.__init__(self, verbose)

        self._ref_probs = None
        self._ref_nrm = None
        self._ref_ft = None
        self._ref_phs_spec = None
        self._ref_mag_spec = None
        self._ref_scorrs = None
        self._ref_asymms_1 = None
        self._ref_asymms_2 = None
        self._ref_ecop_dens_arrs = None
        self._ref_ecop_etpy_arrs = None
        self._ref_nth_ords_cdfs_dict = None
        self._ref_nth_ord_diffs = None
        self._ref_ft_cumm_corr = None
        self._ref_phs_cross_corr_mat = None
        self._ref_cos_sin_dists_dict = None

        # add var labs to _get_sim_data in save.py if then need to be there
        self._sim_probs = None
        self._sim_nrm = None
        self._sim_ft = None
        self._sim_phs_spec = None
        self._sim_mag_spec = None
        self._sim_scorrs = None
        self._sim_asymms_1 = None
        self._sim_asymms_2 = None
        self._sim_ecop_dens_arrs = None
        self._sim_ecop_etpy_arrs = None
        self._sim_nth_ord_diffs = None
        self._sim_shape = None
        self._sim_phs_cross_corr_mat = None
        self._sim_mag_spec_cdf = None

        # a list that holds the indicies of to and from phases to optimize,
        # the total number of classes, the current class index.
        self._sim_phs_ann_class_vars = None

        # An array. False for phas changes, True for coeff changes
        self._sim_mag_spec_flags = None

        self._sim_mag_spec_idxs = None
        self._sim_rltzns_proto_tup = None

        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False

        self._prep_prep_flag = False

        self._prep_verify_flag = False
        return

    def _set_ann_cls_widths(self,):

        n_coeffs = 1 + (self._data_ref_rltzn.size // 2)

        if ((self._sett_ann_phs_ann_class_width is not None) and
            (self._sett_ann_phs_ann_class_width < n_coeffs)):

            phs_ann_clss = int(mceil(
                n_coeffs / self._sett_ann_phs_ann_class_width))

            assert phs_ann_clss > 0

            assert (
                (phs_ann_clss * self._sett_ann_phs_ann_class_width) >=
                n_coeffs)

            self._sim_phs_ann_class_vars = [
                0, self._sett_ann_phs_ann_class_width, phs_ann_clss, 0]

        else:
            self._sett_ann_phs_ann_class_width = n_coeffs
            self._sim_phs_ann_class_vars = [0, n_coeffs, 1, 0]

        return

    def _get_cos_sin_dists_dict(self, ft):

        probs = np.arange(1.0, ft.size + 1) / (ft.size + 1.0)

        out_dict = {}

        cos_vals = np.sort(ft.real)
        sin_vals = np.sort(ft.imag)

        out_dict['cos'] = interp1d(
            cos_vals,
            probs,
            bounds_error=False,
            assume_sorted=True,
            fill_value=(0, 1))

        out_dict['sin'] = interp1d(
            sin_vals,
            probs,
            bounds_error=False,
            assume_sorted=True,
            fill_value=(0, 1))

        return out_dict

    def _get_srtd_nth_diffs_arrs(self, vals):

        assert self._sett_obj_nth_ords is not None, 'nth_ords not defined!'

        srtd_nth_ord_diffs_dict = {}
        for nth_ord in self._sett_obj_nth_ords:

            srtd_nth_ord_diffs_dict[nth_ord] = np.diff(vals, n=nth_ord)

            srtd_nth_ord_diffs_dict[nth_ord] = np.sort(
                srtd_nth_ord_diffs_dict[nth_ord])

        return srtd_nth_ord_diffs_dict

    def _get_nth_ord_diff_cdfs_dict(self, vals):

        diffs_dict = self._get_srtd_nth_diffs_arrs(vals)

        nth_ords_cdfs_dict = {}
        for nth_ord in self._sett_obj_nth_ords:

            diffs = diffs_dict[nth_ord]
            probs = np.linspace(
                1 / diffs.size, 1 - (1 / diffs.size), diffs.size)

            nth_ords_cdfs_dict[nth_ord] = interp1d(
                diffs,
                probs,
                bounds_error=False,
                assume_sorted=True,
                fill_value=(0, 1))

        return nth_ords_cdfs_dict

    def _get_probs(self, data):

        ranks = rankdata(data, method='average')

        probs = ranks / (data.size + 1.0)

        assert np.all((0 < probs) & (probs < 1)), 'probs out of range!'

        return probs

    def _get_probs_norms(self, data):

        probs = self._get_probs(data)

        norms = norm.ppf(probs, loc=0.0, scale=1.0)

        assert np.all(np.isfinite(norms)), 'Invalid values in norms!'

        return probs, norms

    def _get_asymm_1_max(self, scorr):

        a_max = (
            0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / 3.0)))

        return a_max

    def _get_asymm_2_max(self, scorr):

        a_max = (
            0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / 3.0)))

        return a_max

    def _get_etpy_min(self, n_bins):

        dens = 1 / n_bins

        etpy = -np.log(dens)

        return etpy

    def _get_etpy_max(self, n_bins):

        dens = (1 / (n_bins ** 2))

        etpy = -np.log(dens)

        return etpy

    def _get_phs_cross_corr_mat(self, phs_spec):

        n_phas = phs_spec.size

        corr_mat = np.empty((n_phas, n_phas), dtype=float)

        for i in range(n_phas):
            for j in range(n_phas):
                if i <= j:
                    corr_mat[i, j] = np.cos(phs_spec[i] - phs_spec[j])

                else:
                    corr_mat[i, j] = corr_mat[j, i]

        assert np.all((corr_mat >= -1) & (corr_mat <= +1))

        return corr_mat

    def _get_obj_vars(self, probs):

        if (self._sett_obj_scorr_flag or
            self._sett_obj_asymm_type_1_flag or
            self._sett_obj_asymm_type_2_flag):

            scorrs = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            scorrs = None

        if self._sett_obj_asymm_type_1_flag:
            asymms_1 = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            asymms_1 = None

        if self._sett_obj_asymm_type_2_flag:
            asymms_2 = np.full(self._sett_obj_lag_steps.size, np.nan)

        else:
            asymms_2 = None

        if self._sett_obj_ecop_dens_flag or self._sett_obj_ecop_etpy_flag:
            ecop_dens_arrs = np.full(
                (self._sett_obj_lag_steps.size,
                 self._sett_obj_ecop_dens_bins,
                 self._sett_obj_ecop_dens_bins),
                 np.nan,
                 dtype=np.float64)

        else:
            ecop_dens_arrs = None

        if self._sett_obj_ecop_etpy_flag:
            ecop_etpy_arrs = np.full(
                (self._sett_obj_lag_steps.size,),
                 np.nan,
                 dtype=np.float64)

            etpy_min = self._get_etpy_min(self._sett_obj_ecop_dens_bins)
            etpy_max = self._get_etpy_max(self._sett_obj_ecop_dens_bins)

        else:
            ecop_etpy_arrs = etpy_min = etpy_max = None

        if self._sett_obj_nth_ord_diffs_flag:
            nth_ord_diffs = self._get_srtd_nth_diffs_arrs(probs)

        else:
            nth_ord_diffs = None

        if (self._sett_obj_asymm_type_1_flag and
            self._sett_obj_asymm_type_2_flag):

            double_flag = True

        else:
            double_flag = False

        for i, lag in enumerate(self._sett_obj_lag_steps):

            probs_i, rolled_probs_i = roll_real_2arrs(probs, probs, lag)

            if scorrs is not None:
                scorrs[i] = np.corrcoef(probs_i, rolled_probs_i)[0, 1]

            if double_flag:
                asymms_1[i], asymms_2[i] = get_asymms_sample(
                    probs_i, rolled_probs_i)

            else:
                if asymms_1 is not None:
                    asymms_1[i] = get_asymm_1_sample(probs_i, rolled_probs_i)

                if asymms_2 is not None:
                    asymms_2[i] = get_asymm_2_sample(probs_i, rolled_probs_i)

            if asymms_1 is not None:
                asymms_1[i] = asymms_1[i] / self._get_asymm_1_max(scorrs[i])

            if asymms_2 is not None:
                asymms_2[i] = asymms_2[i] / self._get_asymm_2_max(scorrs[i])

            if ecop_dens_arrs is not None:
                fill_bi_var_cop_dens(
                    probs_i, rolled_probs_i, ecop_dens_arrs[i, :, :])

            if ecop_etpy_arrs is not None:
                non_zero_idxs = (ecop_dens_arrs[i, :, :] != 0)

                if non_zero_idxs.sum():
                    dens = ecop_dens_arrs[i][non_zero_idxs]

                    etpy = (-(dens * np.log(dens))).sum()

                    etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

                    assert 0 <= etpy <= 1, 'etpy out of bounds!'

                    ecop_etpy_arrs[i] = etpy

        if scorrs is not None:
            assert np.all(np.isfinite(scorrs)), 'Invalid values in scorrs!'

            assert np.all((scorrs >= -1.0) & (scorrs <= +1.0)), (
                'scorrs out of range!')

        if not self._sett_obj_scorr_flag:
            scorrs = None

#         if asymms_1 is not None:
#             assert np.all(np.isfinite(asymms_1)), 'Invalid values in asymms_1!'
#
#             assert np.all((asymms_1 >= -1.0) & (asymms_1 <= +1.0)), (
#                 'asymms_1 out of range!')

        if asymms_2 is not None:
            assert np.all(np.isfinite(asymms_2)), 'Invalid values in asymms_2!'

            assert np.all((asymms_2 >= -1.0) & (asymms_2 <= +1.0)), (
                'asymms_2 out of range!')

        if ecop_dens_arrs is not None:
            assert np.all(np.isfinite(ecop_dens_arrs)), (
                'Invalid values in ecop_dens_arrs!')

        if ecop_etpy_arrs is not None:
            assert np.all(np.isfinite(ecop_etpy_arrs)), (
                'Invalid values in ecop_etpy_arrs!')

            assert np.all(ecop_etpy_arrs >= 0), (
                'ecop_etpy_arrs values out of range!')

            assert np.all(ecop_etpy_arrs <= 1), (
                'ecop_etpy_arrs values out of range!')

        if nth_ord_diffs is not None:
            for nth_ord in nth_ord_diffs:
                assert np.all(np.isfinite(nth_ord_diffs[nth_ord])), (
                    'Invalid values in nth_ord_diffs!')

        return (
            scorrs,
            asymms_1,
            asymms_2,
            ecop_dens_arrs,
            ecop_etpy_arrs,
            nth_ord_diffs)

    def _get_cumm_ft_corr(self, ref_ft, sim_ft):

        ref_mag = np.abs(ref_ft)
        ref_phs = np.angle(ref_ft)

        sim_mag = np.abs(sim_ft)
        sim_phs = np.angle(sim_ft)

        numr = (
            ref_mag[1:-1] *
            sim_mag[1:-1] *
            np.cos(ref_phs[1:-1] - sim_phs[1:-1]))

        demr = (
            ((ref_mag[1:-1] ** 2).sum() ** 0.5) *
            ((sim_mag[1:-1] ** 2).sum() ** 0.5))

        return np.cumsum(numr) / demr

    def _gen_ref_aux_data(self):

        if self._data_ref_rltzn.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        probs, norms = self._get_probs_norms(self._data_ref_rltzn)

        ft = np.fft.rfft(norms)

        if self._sim_phs_ann_class_vars[2] != 1:
            ft[self._sim_phs_ann_class_vars[1]:] = 0

            data = np.fft.irfft(ft)
            probs, norms = self._get_probs_norms(data)

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'
        assert np.all(np.isfinite(mag_spec)), 'Invalid values in mag_spec!'

        self._ref_probs = probs
        self._ref_nrm = norms

        self._ref_ft = ft
        self._ref_phs_spec = phs_spec
        self._ref_mag_spec = mag_spec

        self._ref_cos_sin_dists_dict = self._get_cos_sin_dists_dict(
            self._ref_ft)

        self._ref_mag_spec_mean = self._ref_mag_spec.mean()

        self._ref_nth_ords_cdfs_dict = self._get_nth_ord_diff_cdfs_dict(probs)

        (scorrs,
         asymms_1,
         asymms_2,
         ecop_dens_arrs,
         ecop_etpy_arrs,
         nth_ord_diffs) = self._get_obj_vars(probs)

        self._ref_scorrs = scorrs
        self._ref_asymms_1 = asymms_1
        self._ref_asymms_2 = asymms_2
        self._ref_ecop_dens_arrs = ecop_dens_arrs
        self._ref_ecop_etpy_arrs = ecop_etpy_arrs
        self._ref_nth_ord_diffs = nth_ord_diffs

        self._ref_ft_cumm_corr = self._get_cumm_ft_corr(
            self._ref_ft, self._ref_ft)

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 1:
            raise NotImplementedError('Implementation for 1D only!')

        if self._sett_extnd_len_set_flag:

            if self._sim_phs_ann_class_vars[2] != 1:
                raise NotImplementedError('Don\'t know how to do this yet!')

            self._sim_shape = (1 +
                ((self._data_ref_shape[0] *
                  self._sett_extnd_len_rel_shp[0]) // 2),)

            if self._sett_obj_sort_init_sim_flag:
                nrm_sort = np.sort(np.concatenate(
                    [self._ref_nrm, ] * self._sett_extnd_len_rel_shp[0]))

                ft = np.fft.rfft(nrm_sort)

                self._sim_mag_spec_flags = np.ones(self._sim_shape, dtype=bool)

            else:
                # randomizing the phase spectrum is necessary.
                # simply taking the ft coeffs of the reference results in
                # optimization producing series that are very similar to the
                # original.
                rands = np.random.random((self._data_ref_shape[0] // 2) - 1)

                phs_spec_base = -np.pi + (2 * np.pi * rands)

                assert np.all(np.isfinite(phs_spec_base)), (
                    'Invalid values in phs_spec_base!')

                ft_base = np.full(
                    (self._data_ref_shape[0] // 2) + 1,
                    np.nan,
                    dtype=np.complex)

                ft_base[+0] = self._ref_ft[+0]
                ft_base[-1] = self._ref_ft[-1]

                ft_base.real[1:-1] = (
                    np.cos(phs_spec_base) * self._ref_mag_spec[1:-1])

                ft_base.imag[1:-1] = (
                    np.sin(phs_spec_base) * self._ref_mag_spec[1:-1])

                ft = np.full(self._sim_shape, np.nan, dtype=np.complex)

                self._sim_mag_spec_flags = np.zeros(
                    self._sim_shape, dtype=bool)

                exps = expon.ppf(
                    np.random.random(self._sim_shape[0]),
                    scale=(self._ref_mag_spec_mean *
                           self._sett_extnd_len_rel_shp[0]))

                exps.sort()

                exps = exps[::-1]

                lst_i = 0
                for i in range(1, 1 + (self._data_ref_shape[0] // 2)):
                    ft[i * self._sett_extnd_len_rel_shp[0]] = ft_base[i]

                    self._sim_mag_spec_flags[
                        i * self._sett_extnd_len_rel_shp[0]] = True

                    # for the rest of the spectrum, generate a random coeff
                    for j in range(lst_i, i * self._sett_extnd_len_rel_shp[0]):

                        assert j != i * self._sett_extnd_len_rel_shp[0], (
                            f'This ({j, i * self._sett_extnd_len_rel_shp[0]}) '
                            f'is not supposed to happen!')

                        rand = exps[j]

                        new_phs = -np.pi + (2 * np.pi * np.random.random())

                        new_coeff = (
                            (rand * np.cos(new_phs)) +
                            (rand * np.sin(new_phs) * 1j))

                        ft[j] = new_coeff

                        self._sim_mag_spec_flags[j] = True

                    lst_i = (i * self._sett_extnd_len_rel_shp[0]) + 1

                ft[+0] = ft_base[+0]
                ft[-1] = ft_base[-1]

                self._sim_mag_spec_flags[+0] = False
                self._sim_mag_spec_flags[-1] = False

        else:
            self._sim_shape = (1 + (self._data_ref_shape[0] // 2),)

            if self._sett_obj_sort_init_sim_flag:
                if self._sim_phs_ann_class_vars[2] != 1:
                    raise NotImplementedError('Don\'t know how to do this yet!')

                nrm_sort = np.sort(self._ref_nrm)

                ft = np.fft.rfft(nrm_sort)

                self._sim_mag_spec_flags = np.ones(self._sim_shape, dtype=bool)

            else:
                if not self._sim_phs_ann_class_vars[3]:
                    rands = np.random.random(self._sim_shape[0])

                    phs_spec = -np.pi + (2 * np.pi * rands)

                    assert np.all(np.isfinite(phs_spec)), (
                        'Invalid values in phs_spec!')

                    ft = np.full(self._sim_shape, np.nan, dtype=np.complex)

                    ft.real[:] = np.cos(phs_spec) * self._ref_mag_spec[:]
                    ft.imag[:] = np.sin(phs_spec) * self._ref_mag_spec[:]

                else:

                    bix, eix = self._sim_phs_ann_class_vars[:2]

                    rands = np.random.random(eix - bix)

                    phs_spec = -np.pi + (2 * np.pi * rands)

                    assert np.all(np.isfinite(phs_spec)), (
                        'Invalid values in phs_spec!')

                    ft = self._sim_ft.copy()

                    ft.real[bix:eix] = (
                        self._ref_mag_spec[bix:eix] * np.cos(rands))

                    ft.imag[bix:eix] = (
                        self._ref_mag_spec[bix:eix] * np.sin(rands))

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'

        data = np.fft.irfft(ft)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        probs, norms = self._get_probs_norms(data)

        self._sim_probs = probs
        self._sim_nrm = norms

        self._sim_ft = ft
        self._sim_phs_spec = np.angle(ft)  # don't use phs_spec from above
        self._sim_mag_spec = np.abs(ft)

        (scorrs,
         asymms_1,
         asymms_2,
         ecop_dens_arrs,
         ecop_etpy_arrs,
         nth_ord_diffs) = self._get_obj_vars(probs)

        self._sim_scorrs = scorrs
        self._sim_asymms_1 = asymms_1
        self._sim_asymms_2 = asymms_2
        self._sim_ecop_dens_arrs = ecop_dens_arrs
        self._sim_ecop_etpy_arrs = ecop_etpy_arrs
        self._sim_nth_ord_diffs = nth_ord_diffs

        if not self._sett_extnd_len_set_flag:
            self._sim_mag_spec_idxs = np.argsort(self._sim_mag_spec[1:])[::-1]

        if self._sett_ann_mag_spec_cdf_idxs_flag:
            mag_spec_pdf = self._sim_mag_spec / self._sim_mag_spec.sum()
            mag_spec_cdf = np.concatenate(([0], np.cumsum(mag_spec_pdf)))

            self._sim_mag_spec_cdf = interp1d(
                mag_spec_cdf,
                np.arange(mag_spec_cdf.size),
                bounds_error=True,
                assume_sorted=True)

        self._prep_sim_aux_flag = True
        return

    def prepare(self):

        '''Generate data required before phase annealing starts'''

        PAS._PhaseAnnealingSettings__verify(self)
        assert self._sett_verify_flag, 'Settings in an unverfied state!'

        self._set_ann_cls_widths()

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

        sim_rltzns_out_labs = [
            'ft',
            'probs',
            'nrm',
            'scorrs',
            'asymms_1',
            'asymms_2',
            'ecop_dens',
            'ecop_entps',
            'iter_ctr',
            'iters_wo_acpt',
            'tol',
            'fin_temp',
            'stopp_criteria',
            'tols',
            'obj_vals_all',
            'acpts_rjts_all',
            'acpt_rates_all',
            'obj_vals_min',
            'phss_all',
            'temps',
            'phs_red_rates',
            'idxs_all',
            'idxs_acpt',
            'acpt_rates_dfrntl',
            'ft_cumm_corr_sim_ref',
            'ft_cumm_corr_sim_sim',
            'phs_cross_corr_mat',
            ]

        sim_rltzns_out_labs.extend(
            [f'sim_nth_ord_diffs_{nth_ord:03d}'
             for nth_ord in self._sett_obj_nth_ords])

        self._sim_rltzns_proto_tup = namedtuple(
            'SimRltznData', sim_rltzns_out_labs)

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify
