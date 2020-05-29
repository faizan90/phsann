'''
Created on Dec 27, 2019

@author: Faizan
'''
from math import ceil as mceil
from collections import namedtuple
from itertools import combinations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rankdata, expon

from ..misc import print_sl, print_el, roll_real_2arrs
from ..cyth import (
    get_asymms_sample,
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    )

from .settings import PhaseAnnealingSettings as PAS

extrapolate_flag = False
exterp_fil_vals = (0, 1)


class PhaseAnnealingPrepareTfms:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.
    '''

    def _get_probs(self, data, make_like_ref_flag=False):

        probs_all = np.empty_like(data, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            probs = rankdata(data[:, i], method='average')
            probs /= data.shape[0] + 1.0

            if make_like_ref_flag:
                assert self._ref_probs_srtd is not None

                probs = self._ref_probs_srtd[np.argsort(np.argsort(probs)), i]

            assert np.all((0 < probs) & (probs < 1)), 'probs out of range!'

            probs_all[:, i] = probs

        return probs_all

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

    def _get_cumm_ft_corr(self, ref_ft, sim_ft):

        ref_mag = np.abs(ref_ft)
        ref_phs = np.angle(ref_ft)

        sim_mag = np.abs(sim_ft)
        sim_phs = np.angle(sim_ft)

        numr = (
            ref_mag[1:-1, :] *
            sim_mag[1:-1, :] *
            np.cos(ref_phs[1:-1, :] - sim_phs[1:-1, :]))

        demr = (
            ((ref_mag[1:-1, :] ** 2).sum(axis=0) ** 0.5) *
            ((sim_mag[1:-1, :] ** 2).sum(axis=0) ** 0.5))

        return np.cumsum(numr, axis=0) / demr

    def _get_sim_ft_pln(self, rnd_mag_flag=False):

        if not self._sim_phs_ann_class_vars[3]:
            ft = np.zeros(self._sim_shape, dtype=np.complex)

        else:
            ft = self._sim_ft.copy()

        bix, eix = self._sim_phs_ann_class_vars[:2]

        rands = np.random.random((eix - bix, 1))

        phs_spec = self._ref_phs_spec[bix:eix, :].copy()
        phs_spec += 1.0 * (-np.pi + (2 * np.pi * rands))  # out of bound phs

        if rnd_mag_flag:

            raise NotImplementedError('Not for mult cols yet!')
            # TODO: sample based on PDF?
            # TODO: Could also be done by rearranging based on ref_mag_spec
            # order.

            # Assuming that the mag_spec follows an expon dist.
            mag_spec = expon.ppf(
                np.random.random(self._sim_shape),
                scale=(self._ref_mag_spec_mean *
                       self._sett_extnd_len_rel_shp[0]))

            mag_spec.sort(axis=0)

            mag_spec = mag_spec[::-1, :]

            mag_spec_flags = np.zeros(mag_spec.shape, dtype=bool)

            mag_spec_flags[bix:eix + 1, :] = True

        else:
            mag_spec = self._ref_mag_spec.copy()

            mag_spec_flags = None

        ft.real[bix:eix, :] = mag_spec[bix:eix, :] * np.cos(phs_spec)
        ft.imag[bix:eix, :] = mag_spec[bix:eix, :] * np.sin(phs_spec)

        self._sim_phs_mod_flags[bix:eix, :] += 1

        return ft, mag_spec_flags


class PhaseAnnealingPrepareCDFS:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.

    NOTE: Add CDF ftns to _trunc_interp_ftns for trunction.
    '''

    def _get_data_ft(self, data, vtype, norm_vals):

        data_ft = np.fft.rfft(data, axis=0)
        data_mag_spec = np.abs(data_ft)[1:-1]

        data_mag_spec = (data_mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (norm_vals is not None):
            data_mag_spec /= norm_vals

        elif (vtype == 'ref') and (norm_vals is None):
            self._ref_data_ft_norm_vals = data_mag_spec[-1, :].copy()

            data_mag_spec /= self._ref_data_ft_norm_vals

        else:
            raise NotImplementedError

        return data_mag_spec

    def _get_mult_ecop_dens_diffs_cdfs_dict(self, probs):

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        ecop_dens_arr = np.full(
            (self._sett_obj_ecop_dens_bins,
             self._sett_obj_ecop_dens_bins),
            np.nan,
            dtype=np.float64)

        out_dict = {}
        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            for comb in combs:
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Ecop density differences configured for pairs only!')

                fill_bi_var_cop_dens(
                    probs[:, col_idxs[0]],
                    probs[:, col_idxs[1]],
                    ecop_dens_arr)

#                 srtd_ecop_dens = np.sort(ecop_dens_arr.ravel())
#
#                 cdf_vals = np.arange(
#                     1.0, (self._sett_obj_ecop_dens_bins ** 2) + 1) / (
#                     (self._sett_obj_ecop_dens_bins ** 2) + 1.0)
#
#                 if not extrapolate_flag:
#                     interp_ftn = interp1d(
#                         srtd_ecop_dens,
#                         cdf_vals,
#                         bounds_error=False,
#                         assume_sorted=True,
#                         fill_value=exterp_fil_vals)
#
#                 else:
#                     interp_ftn = interp1d(
#                         srtd_ecop_dens,
#                         cdf_vals,
#                         bounds_error=False,
#                         assume_sorted=True,
#                         fill_value='extrapolate',
#                         kind='slinear')
#
#                 assert not hasattr(interp_ftn, 'wts')
#                 assert not hasattr(interp_ftn, 'sclr')
#
#                 wts = (1 / (cdf_vals.size + 1)) / (1 - cdf_vals)
#
#                 sclr = (cdf_vals.size / ecop_dens_arr.size)
#
#                 interp_ftn.wts = wts
#                 interp_ftn.sclr = sclr
#
#                 out_dict[comb] = interp_ftn
                out_dict[comb] = ecop_dens_arr.copy()
        return out_dict

    def _get_mult_asymm_1_diffs_cdfs_dict(self, probs):

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        cdf_vals = np.arange(1.0, probs.shape[0] + 1)
        cdf_vals /= cdf_vals.size + 1.0

        cdf_vals = np.concatenate((
            [0.0],
            cdf_vals,
            [1.0],
            ))

        wts = (1 / (cdf_vals.size - 2)) / (
            (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

        out_dict = {}
        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            for comb in combs:
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 1 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] + probs[:, col_idxs[1]] - 1.0) ** 3)

                diff_vals = np.concatenate((
                    [-1],
                    diff_vals,
                    [+1]))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

                interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diff_vals) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[comb] = interp_ftn

        return out_dict

    def _get_mult_asymm_2_diffs_cdfs_dict(self, probs):

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        cdf_vals = np.arange(1.0, probs.shape[0] + 1)
        cdf_vals /= cdf_vals.size + 1.0

        cdf_vals = np.concatenate((
            [0.0],
            cdf_vals,
            [1.0],
            ))

        wts = (1 / (cdf_vals.size - 2)) / (
            (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

        out_dict = {}
        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            for comb in combs:
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] - probs[:, col_idxs[1]]) ** 3)

                diff_vals = np.concatenate((
                    [-1],
                    diff_vals,
                    [+1]))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

                interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diff_vals) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[comb] = interp_ftn

        return out_dict

    def _get_pcorr_diffs_cdfs_dict(self, data):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                data_i, rolled_data_i = roll_real_2arrs(
                    data[:, i], data[:, i], lag)

                cdf_vals = np.arange(1.0, data_i.size + 1)
                cdf_vals /= cdf_vals.size + 1.0

                cdf_vals = np.concatenate((
                    [0.0],
                    cdf_vals,
                    [1.0],
                    ))

                diff_vals = np.sort((rolled_data_i - data_i))

                diff_vals = np.concatenate((
                    [data[:, i].min() - data[:, i].max()],
                    diff_vals,
                    [data[:, i].max() - data[:, i].min()]))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

                wts = (1 / (cdf_vals.size - 2)) / (
                    (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

                interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diff_vals) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_scorr_diffs_cdfs_dict(self, probs):

        out_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                cdf_vals = np.arange(1.0, probs_i.size + 1.0)
                cdf_vals /= cdf_vals.size + 1.0

                cdf_vals = np.concatenate((
                    [0.0],
                    cdf_vals,
                    [1.0],
                    ))

                diff_vals = np.sort(rolled_probs_i * probs_i)

                diff_vals = np.concatenate((
                    [+0],
                    diff_vals,
                    [+1]))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

                wts = (1 / (cdf_vals.size - 2)) / (
                    (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

#                 diff_vals_med = np.median(diff_vals[1:-1])
#                 wts = np.abs(diff_vals[1:-1] - diff_vals_med)
#                 wts /= wts.max()

                # Squared because original wts. had a product in it.
#                 interp_ftn.wts = wts ** 2

                interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diff_vals) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_asymm_1_diffs_cdfs_dict(self, probs):

        out_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                cdf_vals = np.arange(1.0, probs_i.size + 1)
                cdf_vals /= cdf_vals.size + 1.0

#                 cdf_vals = np.concatenate((
#                     [0.0],
#                     cdf_vals,
#                     [1.0],
#                     ))
#
#                 diff_vals = np.concatenate((
#                     [-1],
#                     np.sort((probs_i + rolled_probs_i - 1.0) ** 7),
#                     [+1]))

                diff_vals = np.sort((probs_i + rolled_probs_i - 1.0) ** 7)

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

#                 wts = (1 / (cdf_vals.size - 2)) / (
#                     (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

                wts = (1 / (cdf_vals.size - 2)) / (
                    (cdf_vals * (1 - cdf_vals)))

                interp_ftn.wts = wts

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_asymm_2_diffs_cdfs_dict(self, probs):

        out_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                cdf_vals = np.arange(1.0, probs_i.size + 1)
                cdf_vals /= cdf_vals.size + 1.0

                cdf_vals = np.concatenate((
                    [0.0],
                    cdf_vals,
                    [1.0],
                    ))

                diff_vals = np.concatenate((
                    [-1],
                    np.sort((probs_i - rolled_probs_i) ** 3),
                    [+1]))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        diff_vals,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')

                wts = (1 / (cdf_vals.size - 2)) / (
                    (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

#                 diff_vals_med = np.median(diff_vals[1:-1])
#                 wts = np.abs(diff_vals[1:-1] - diff_vals_med)
#                 wts /= wts.max()

                # Squared because original wts. had a product in it.
#                 interp_ftn.wts = wts ** 2

                interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diff_vals) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_ecop_dens_diffs_cdfs_dict(self, probs):

        out_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                ecop_dens_arr = np.full(
                    (self._sett_obj_ecop_dens_bins,
                     self._sett_obj_ecop_dens_bins),
                    np.nan,
                    dtype=np.float64)

                fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arr)

                srtd_ecop_dens = np.sort(ecop_dens_arr.ravel())

                srtd_ecop_dens = np.concatenate((
                    [+0],
                    srtd_ecop_dens,
                    [+1]))

                cdf_vals = np.arange(
                    1.0, (self._sett_obj_ecop_dens_bins ** 2) + 1) / (
                    (self._sett_obj_ecop_dens_bins ** 2) + 1.0)

                cdf_vals = np.concatenate((
                    [0.0],
                    cdf_vals,
                    [1.0],
                    ))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        srtd_ecop_dens,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        srtd_ecop_dens,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')
                assert not hasattr(interp_ftn, 'sclr')

                wts = (1 / (cdf_vals.size - 2)) / (1 - cdf_vals[1:-1])

                sclr = (cdf_vals.size / ecop_dens_arr.size)

                interp_ftn.wts = wts
                interp_ftn.sclr = sclr

#                 exct_diffs = interp_ftn(srtd_ecop_dens) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_ecop_etpy_diffs_cdfs_dict(self, probs):

        out_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                ecop_dens_arr = np.full(
                    (self._sett_obj_ecop_dens_bins,
                     self._sett_obj_ecop_dens_bins),
                    np.nan,
                    dtype=np.float64)

                fill_bi_var_cop_dens(probs_i, rolled_probs_i, ecop_dens_arr)

                ecop_dens_arr = ecop_dens_arr.ravel()

                non_zero_idxs = ecop_dens_arr > 0

                dens = ecop_dens_arr[non_zero_idxs]

                etpy = (-(dens * np.log(dens)))

                etpys_arr = np.zeros_like(ecop_dens_arr)

                etpys_arr[non_zero_idxs] = etpy

                # FIXME: the zeros in the array have too much weight.
                srtd_etpys_arr = np.sort(etpys_arr)

                srtd_etpys_arr = np.concatenate((
                    [+0],
                    srtd_etpys_arr,
                    [+1]))

                cdf_vals = np.arange(
                    1.0, (self._sett_obj_ecop_dens_bins ** 2) + 1) / (
                    (self._sett_obj_ecop_dens_bins ** 2) + 1.0)

                cdf_vals = np.concatenate((
                    [0.0],
                    cdf_vals,
                    [1.0],
                    ))

                if not extrapolate_flag:
                    interp_ftn = interp1d(
                        srtd_etpys_arr,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value=exterp_fil_vals)

                else:
                    interp_ftn = interp1d(
                        srtd_etpys_arr,
                        cdf_vals,
                        bounds_error=False,
                        assume_sorted=True,
                        fill_value='extrapolate',
                        kind='slinear')

                assert not hasattr(interp_ftn, 'wts')
                assert not hasattr(interp_ftn, 'sclr')

                wts = (1 / (cdf_vals.size - 2)) / (1 - cdf_vals[1:-1])

                sclr = probs_i.size / (ecop_dens_arr.size)

                interp_ftn.wts = wts
                interp_ftn.sclr = sclr

#                 exct_diffs = interp_ftn(srtd_etpys_arr) - cdf_vals
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

                out_dict[(label, lag)] = interp_ftn

        return out_dict

    def _get_cos_sin_cdfs_dict(self, ft):

        out_dict = {}
        cdf_vals = np.arange(1.0, ft.shape[0] + 1.0) / (ft.shape[0] + 1.0)

        cdf_vals = np.concatenate((
            [0.0],
            cdf_vals,
            [1.0],
            ))

        for i, label in enumerate(self._data_ref_labels):
            cos_vals = np.sort(ft.real[:, i])
            sin_vals = np.sort(ft.imag[:, i])

            cos_vals = np.concatenate((
                [-self._ref_mag_spec.max()],
                cos_vals,
                [+self._ref_mag_spec.max()]))

            sin_vals = np.concatenate((
                [-self._ref_mag_spec.max()],
                sin_vals,
                [+self._ref_mag_spec.max()]))

            if not extrapolate_flag:
                out_dict[(label, 'cos')] = interp1d(
                    cos_vals,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=exterp_fil_vals)

                out_dict[(label, 'sin')] = interp1d(
                    sin_vals,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value=exterp_fil_vals)

            else:
                out_dict[(label, 'cos')] = interp1d(
                    cos_vals,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value='extrapolate',
                    kind='slinear')

                out_dict[(label, 'sin')] = interp1d(
                    sin_vals,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value='extrapolate',
                    kind='slinear')

            wts = (1 / (cdf_vals.size - 2)) / (
                (cdf_vals[1:-1] * (1 - cdf_vals[1:-1])))

            out_dict[(label, 'cos')].wts = wts
            out_dict[(label, 'sin')].wts = out_dict[(label, 'cos')].wts

            out_dict[(label, 'cos')].sclr = 2
            out_dict[(label, 'sin')].sclr = out_dict[(label, 'cos')].sclr

#             exct_cos_diffs = out_dict[(label, 'cos')](cos_vals) - probs
#             exct_sin_diffs = out_dict[(label, 'sin')](sin_vals) - probs
#
#             assert np.all(np.isclose(exct_cos_diffs, 0.0)), (
#                 'Interpolation function not keeping best estimates!')
#
#             assert np.all(np.isclose(exct_sin_diffs, 0.0)), (
#                 'Interpolation function not keeping best estimates!')

        return out_dict

    def _get_srtd_nth_diffs_arrs(self, vals, nth_ords):

        assert self._sett_obj_nth_ords is not None, 'nth_ords not defined!'

        srtd_nth_ord_diffs_dict = {}

        for i, label in enumerate(self._data_ref_labels):
            for nth_ord in nth_ords:

                diffs = np.diff(vals[:, i], n=nth_ord)

                srtd_nth_ord_diffs_dict[(label, nth_ord)] = np.sort(diffs)

        return srtd_nth_ord_diffs_dict

    def _get_nth_ord_diff_cdfs_dict(self, vals, nth_ords):

        diffs_dict = self._get_srtd_nth_diffs_arrs(vals, nth_ords)

        nth_ords_cdfs_dict = {}

        wts_exp = 1.0

        for lab_nth_ord, diffs in diffs_dict.items():
#             nth_ord = lab_nth_ord[1]

            cdf_vals = np.arange(1.0, diffs.size + 1.0)
            cdf_vals /= cdf_vals.size + 1.0

#             cdf_vals = np.concatenate((
#                 [0.0],
#                 cdf_vals,
#                 [1.0],
#                 ))
#
#             diffs = np.concatenate((
#                 [-nth_ord * 2],
#                 diffs,
#                 [+nth_ord * 2]))

            if not extrapolate_flag:
                interp_ftn = interp1d(
                    diffs,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=False,
                    fill_value=exterp_fil_vals)

            else:
                interp_ftn = interp1d(
                    diffs,
                    cdf_vals,
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value='extrapolate',
                    kind='slinear')

            assert not hasattr(interp_ftn, 'wts')

#             wts = ((1 / (cdf_vals.size - 2)) ** wts_exp) / (
#                 ((cdf_vals[1:-1] ** wts_exp) *
#                  ((1 - cdf_vals[1:-1]) ** wts_exp)))

            wts = ((1 / (cdf_vals.size - 1)) ** wts_exp) / (
                ((cdf_vals ** wts_exp) *
                 ((1 - cdf_vals) ** wts_exp)))

            interp_ftn.wts = wts

#                 exct_diffs = interp_ftn(diffs) - probs
#
#                 assert np.all(np.isclose(exct_diffs, 0.0)), (
#                     'Interpolation function not keeping best estimates!')

            nth_ords_cdfs_dict[lab_nth_ord] = interp_ftn

        return nth_ords_cdfs_dict


class PhaseAnnealingPrepare(
        PAS,
        PhaseAnnealingPrepareTfms,
        PhaseAnnealingPrepareCDFS):

    '''Prepare derived variables required by phase annealing here'''

    def __init__(self, verbose=True):
        # TODO: Some of these have to be reset when switching to a phase class.

        PAS.__init__(self, verbose)

        # Reference.
        self._ref_probs = None
        self._ref_ft = None
        self._ref_phs_spec = None
        self._ref_mag_spec = None
        self._ref_scorrs = None
        self._ref_asymms_1 = None
        self._ref_asymms_2 = None
        self._ref_ecop_dens = None
        self._ref_ecop_etpy = None
        self._ref_ft_cumm_corr = None
        self._ref_phs_ann_class_vars = None
        self._ref_phs_ann_n_clss = None
        self._ref_probs_srtd = None
        self._ref_data = None
        self._ref_pcorrs = None
        self._ref_nths = None
        self._ref_data_ft = None
        self._ref_data_ft_norm_vals = None

        self._ref_scorr_diffs_cdfs_dict = None
        self._ref_asymm_1_diffs_cdfs_dict = None
        self._ref_asymm_2_diffs_cdfs_dict = None
        self._ref_ecop_dens_diffs_cdfs_dict = None
        self._ref_ecop_etpy_diffs_cdfs_dict = None
        self._ref_nth_ord_diffs_cdfs_dict = None
        self._ref_pcorr_diffs_cdfs_dict = None

        self._ref_mult_asymm_1_diffs_cdfs_dict = None
        self._ref_mult_asymm_2_diffs_cdfs_dict = None
        self._ref_mult_ecop_dens_diffs_cdfs_dict = None

        # Simulation.
        # Add var labs to _get_sim_data in save.py if then need to be there.
        self._sim_probs = None
        self._sim_ft = None
        self._sim_phs_spec = None
        self._sim_mag_spec = None
        self._sim_scorrs = None
        self._sim_asymms_1 = None
        self._sim_asymms_2 = None
        self._sim_ecop_dens = None
        self._sim_ecop_etpy = None
        self._sim_nth_ord_diffs = None
        self._sim_shape = None
        self._sim_mag_spec_cdf = None
        self._sim_data = None
        self._sim_pcorrs = None
        self._sim_nths = None
        self._sim_ft_best = None
        self._sim_data_ft = None

        # A list that holds the indicies of to and from phases to optimize,
        # the total number of classes and the current class index.
        self._sim_phs_ann_class_vars = None
        self._sim_phs_ann_n_clss = None
        self._sim_phs_mod_flags = None
        self._sim_n_idxs_all_cts = None
        self._sim_n_idxs_acpt_cts = None

        # An array. False for phase changes, True for coeff changes
        self._sim_mag_spec_flags = None

        self._sim_scorr_diffs = None
        self._sim_asymm_1_diffs = None
        self._sim_asymm_2_diffs = None
        self._sim_ecop_dens_diffs = None
        self._sim_ecop_etpy_diffs = None

        self._sim_mult_asymms_1_diffs = None
        self._sim_mult_asymms_2_diffs = None
        self._sim_mult_ecops_dens_diffs = None

        self._sim_mag_spec_idxs = None
        self._sim_rltzns_proto_tup = None

        # Flags.
        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False
        self._prep_prep_flag = False
        self._prep_verify_flag = False

        # Validation steps.
        self._prep_vld_flag = False
        return

    def _set_phs_ann_cls_vars_ref(self):

        # Second index value in _ref_phs_ann_n_clss not inclusive.
        n_coeffs = (self._data_ref_shape[0] // 2) - 1

        if ((self._sett_ann_phs_ann_class_width is not None) and
            (self._sett_ann_phs_ann_class_width < n_coeffs)):

            phs_ann_clss = int(mceil(
                n_coeffs / self._sett_ann_phs_ann_class_width))

            assert phs_ann_clss > 0

            assert (
                (phs_ann_clss * self._sett_ann_phs_ann_class_width) >=
                n_coeffs)

#             phs_ann_class_vars = [
#                 1, self._sett_ann_phs_ann_class_width, phs_ann_clss, 0]

            phs_ann_class_vars = [
                n_coeffs + 1 - self._sett_ann_phs_ann_class_width,
                n_coeffs + 1,
                phs_ann_clss,
                0]

        else:
            phs_ann_class_vars = [1, n_coeffs + 1, 1, 0]

        self._ref_phs_ann_class_vars = np.array(phs_ann_class_vars, dtype=int)

        self._sett_ann_phs_ann_class_width = (
            self._ref_phs_ann_class_vars[1] - self._ref_phs_ann_class_vars[0])

        self._ref_phs_ann_n_clss = int(self._ref_phs_ann_class_vars[2])
        return

    def _set_phs_ann_cls_vars_sim(self):

        # Assuming _set_phs_ann_cls_vars_ref has been called before.
        # Second index value in _sim_phs_ann_n_clss not inclusive.

        assert self._ref_phs_ann_class_vars is not None, (
            '_ref_phs_ann_class_vars not set!')

        phs_ann_class_vars = self._ref_phs_ann_class_vars.copy()

        phs_ann_class_vars[1] *= self._sett_extnd_len_rel_shp[0]

        self._sim_phs_ann_class_vars = np.array(phs_ann_class_vars, dtype=int)

        self._sim_phs_ann_n_clss = int(self._sim_phs_ann_class_vars[2])
        return

    def _update_obj_vars(self, vtype):

        '''Required variables e.g. self._XXX_probs should have been
        defined/updated before entering.
        '''

        if vtype == 'ref':
            probs = self._ref_probs
            data = self._ref_data

        elif vtype == 'sim':
            probs = self._sim_probs
            data = self._sim_data

        else:
            raise ValueError(f'Unknown vtype in _update_obj_vars: {vtype}!')

        if self._prep_vld_flag:
            lag_steps = self._sett_obj_lag_steps_vld
            nth_ords = self._sett_obj_nth_ords_vld

        else:
            lag_steps = self._sett_obj_lag_steps
            nth_ords = self._sett_obj_nth_ords

        if (self._sett_obj_scorr_flag or
            self._sett_obj_asymm_type_1_flag or
            self._sett_obj_asymm_type_2_flag):

            scorrs = np.full(
                (self._data_ref_n_labels, lag_steps.size),
                np.nan)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                scorr_diffs = {}

            else:
                scorr_diffs = None

        else:
            scorrs = None
            scorr_diffs = None

        if self._sett_obj_pcorr_flag:
            pcorrs = np.full(
                (self._data_ref_n_labels, lag_steps.size),
                np.nan)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                pcorr_diffs = {}

            else:
                pcorr_diffs = None

        else:
            pcorrs = None
            pcorr_diffs = None

        if self._sett_obj_asymm_type_1_flag:
            asymms_1 = np.full(
                (self._data_ref_n_labels, lag_steps.size),
                np.nan)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                asymm_1_diffs = {}

            else:
                asymm_1_diffs = None

        else:
            asymms_1 = None
            asymm_1_diffs = None

        if self._sett_obj_asymm_type_2_flag:
            asymms_2 = np.full(
                (self._data_ref_n_labels, lag_steps.size),
                np.nan)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                asymm_2_diffs = {}

            else:
                asymm_2_diffs = None

        else:
            asymms_2 = None
            asymm_2_diffs = None

        if self._sett_obj_ecop_dens_flag or self._sett_obj_ecop_etpy_flag:
            ecop_dens_arrs = np.full(
                (self._data_ref_n_labels,
                 lag_steps.size,
                 self._sett_obj_ecop_dens_bins,
                 self._sett_obj_ecop_dens_bins),
                np.nan,
                dtype=np.float64)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                ecop_dens_diffs = {}

            else:
                ecop_dens_diffs = None

        else:
            ecop_dens_arrs = None
            ecop_dens_diffs = None

        if self._sett_obj_ecop_etpy_flag:
            ecop_etpy_arrs = np.full(
                (self._data_ref_n_labels, lag_steps.size,),
                np.nan,
                dtype=np.float64)

            etpy_min = self._get_etpy_min(self._sett_obj_ecop_dens_bins)
            etpy_max = self._get_etpy_max(self._sett_obj_ecop_dens_bins)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                ecop_etpy_diffs = {}

            else:
                ecop_etpy_diffs = None

        else:
            ecop_etpy_arrs = etpy_min = etpy_max = None
            ecop_etpy_diffs = None

        if self._sett_obj_nth_ord_diffs_flag:
            nths = np.full((self._data_ref_n_labels, nth_ords.size), np.nan)

            nth_ord_diffs = self._get_srtd_nth_diffs_arrs(probs, nth_ords)

            for j, label in enumerate(self._data_ref_labels):
                for i, nth_ord in enumerate(nth_ords):
                    nths[j, i] = np.sum(nth_ord_diffs[(label, nth_ord)])

            if not self._sett_obj_use_obj_dist_flag:
                nth_ord_diffs = None

        else:
            nth_ord_diffs = None
            nths = None

        if (self._sett_obj_asymm_type_1_flag and
            self._sett_obj_asymm_type_2_flag):

            double_flag = True

        else:
            double_flag = False

        if ((vtype == 'sim') and
            (self._ref_mult_asymm_1_diffs_cdfs_dict is not None)):

            mult_asymm_1_diffs = {}

        else:
            mult_asymm_1_diffs = None

        if ((vtype == 'sim') and
            (self._ref_mult_asymm_2_diffs_cdfs_dict is not None)):

            mult_asymm_2_diffs = {}

        else:
            mult_asymm_2_diffs = None

        if ((vtype == 'sim') and
            (self._ref_mult_ecop_dens_diffs_cdfs_dict is not None)):

            mult_ecop_dens_arr = np.full(
                (self._sett_obj_ecop_dens_bins,
                 self._sett_obj_ecop_dens_bins),
                np.nan,
                dtype=np.float64)

            mult_ecop_dens_diffs = {}

        else:
            mult_ecop_dens_diffs = None

        if self._sett_obj_match_data_ft_flag:
            if vtype == 'sim':
                data_ft_norm_vals = self._ref_data_ft_norm_vals

            else:
                data_ft_norm_vals = None

            data_ft = self._get_data_ft(data, vtype, data_ft_norm_vals)

        else:
            data_ft = None

        for j, label in enumerate(self._data_ref_labels):
            for i, lag in enumerate(lag_steps):

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, j], probs[:, j], lag)

                if scorrs is not None:
                    scorrs[j, i] = np.corrcoef(probs_i, rolled_probs_i)[0, 1]

                    if scorr_diffs is not None:
                        scorr_diffs[(label, lag)] = np.sort(
                            rolled_probs_i * probs_i)

                if double_flag:
                    asymms_1[j, i], asymms_2[j, i] = get_asymms_sample(
                        probs_i, rolled_probs_i)

                    if asymm_1_diffs is not None:
                        asymm_1_diffs[(label, lag)] = np.sort(
                            (probs_i + rolled_probs_i - 1.0) ** 7)

                    if asymm_2_diffs is not None:
                        asymm_2_diffs[(label, lag)] = np.sort(
                            (probs_i - rolled_probs_i) ** 3)

                else:
                    if asymms_1 is not None:
                        asymms_1[j, i] = get_asymm_1_sample(
                            probs_i, rolled_probs_i)

                        if asymm_1_diffs is not None:
                            asymm_1_diffs[(label, lag)] = np.sort(
                                (probs_i + rolled_probs_i - 1.0) ** 7)

                    if asymms_2 is not None:
                        asymms_2[j, i] = get_asymm_2_sample(
                            probs_i, rolled_probs_i)

                        if asymm_2_diffs is not None:
                            asymm_2_diffs[(label, lag)] = np.sort(
                                (probs_i - rolled_probs_i) ** 3)

                if asymms_1 is not None:
                    asymms_1[j, i] /= self._get_asymm_1_max(scorrs[j, i])

                if asymms_2 is not None:
                    asymms_2[j, i] /= self._get_asymm_2_max(scorrs[j, i])

                if ecop_dens_arrs is not None:
                    fill_bi_var_cop_dens(
                        probs_i, rolled_probs_i, ecop_dens_arrs[j, i, :, :])

                    if ecop_dens_diffs is not None:
                        ecop_dens_diffs[(label, lag)] = np.sort(
                            ecop_dens_arrs[j, i, :, :].ravel())

                if ecop_etpy_arrs is not None:
                    non_zero_idxs = ecop_dens_arrs[j, i, :, :] > 0

                    dens = ecop_dens_arrs[j, i][non_zero_idxs]

                    etpy_arr = -(dens * np.log(dens))

                    etpy = etpy_arr.sum()

                    etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

                    assert 0 <= etpy <= 1, 'etpy out of bounds!'

                    ecop_etpy_arrs[j, i] = etpy

                    if ecop_etpy_diffs is not None:
                        etpy_diffs = np.zeros(
                            self._sett_obj_ecop_dens_bins ** 2)

                        etpy_diffs[non_zero_idxs.ravel()] = etpy_arr

                        ecop_etpy_diffs[(label, lag)] = np.sort(etpy_diffs)

                if pcorrs is not None:
                    data_i, rolled_data_i = roll_real_2arrs(
                        data[:, j], data[:, j], lag)

                    pcorrs[j, i] = np.corrcoef(
                        data_i, rolled_data_i)[0, 1]

                    if pcorr_diffs is not None:
                        pcorr_diffs[(label, lag)] = np.sort(
                            (rolled_data_i - data_i))

        if mult_asymm_1_diffs is not None:
            for comb in self._ref_mult_asymm_1_diffs_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 1 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] + probs[:, col_idxs[1]] - 1.0) ** 3)

                mult_asymm_1_diffs[comb] = diff_vals

        if mult_asymm_2_diffs is not None:
            for comb in self._ref_mult_asymm_2_diffs_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] - probs[:, col_idxs[1]]) ** 3)

                mult_asymm_2_diffs[comb] = diff_vals

        if mult_ecop_dens_diffs is not None:
            for comb in self._ref_mult_ecop_dens_diffs_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Ecop density configured for pairs only!')

                fill_bi_var_cop_dens(
                    probs[:, col_idxs[0]],
                    probs[:, col_idxs[1]],
                    mult_ecop_dens_arr)

                mult_ecop_dens_diffs[comb] = mult_ecop_dens_arr.copy()

        if scorrs is not None:
            assert np.all(np.isfinite(scorrs)), 'Invalid values in scorrs!'

            assert np.all((scorrs >= -1.0) & (scorrs <= +1.0)), (
                'scorrs out of range!')

        if not self._sett_obj_scorr_flag:
            scorrs = None

        if asymms_1 is not None:
            assert np.all(np.isfinite(asymms_1)), 'Invalid values in asymms_1!'

            assert np.all((asymms_1 >= -1.0) & (asymms_1 <= +1.0)), (
                'asymms_1 out of range!')

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
            for lab_nth_ord in nth_ord_diffs:
                assert np.all(np.isfinite(nth_ord_diffs[lab_nth_ord])), (
                    'Invalid values in nth_ord_diffs!')

        if pcorrs is not None:
            assert np.all(np.isfinite(pcorrs)), 'Invalid values in pcorrs!'

        if vtype == 'ref':
            self._ref_scorrs = scorrs
            self._ref_asymms_1 = asymms_1
            self._ref_asymms_2 = asymms_2
            self._ref_ecop_dens = ecop_dens_arrs
            self._ref_ecop_etpy = ecop_etpy_arrs
            self._ref_pcorrs = pcorrs
            self._ref_nths = nths
            self._ref_data_ft = data_ft

        elif vtype == 'sim':
            self._sim_scorrs = scorrs
            self._sim_asymms_1 = asymms_1
            self._sim_asymms_2 = asymms_2
            self._sim_ecop_dens = ecop_dens_arrs
            self._sim_ecop_etpy = ecop_etpy_arrs
            self._sim_pcorrs = pcorrs
            self._sim_nths = nths
            self._sim_data_ft = data_ft

            self._sim_scorr_diffs = scorr_diffs
            self._sim_asymm_1_diffs = asymm_1_diffs
            self._sim_asymm_2_diffs = asymm_2_diffs
            self._sim_ecop_dens_diffs = ecop_dens_diffs
            self._sim_ecop_etpy_diffs = ecop_etpy_diffs
            self._sim_nth_ord_diffs = nth_ord_diffs
            self._sim_pcorr_diffs = pcorr_diffs

            self._sim_mult_asymms_1_diffs = mult_asymm_1_diffs
            self._sim_mult_asymms_2_diffs = mult_asymm_2_diffs
            self._sim_mult_ecops_dens_diffs = mult_ecop_dens_diffs

        else:
            raise ValueError(f'Unknown vtype in _update_obj_vars: {vtype}!')

        return

    def _gen_ref_aux_data(self):

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        probs = self._get_probs(self._data_ref_rltzn, False)

        ft = np.fft.rfft(probs, axis=0)

        # FIXME: don't know where to take mean exactly.
        self._ref_mag_spec_mean = (np.abs(ft)).mean(axis=0)

        if self._ref_phs_ann_class_vars[2] != 1:
#             ft[self._ref_phs_ann_class_vars[1]:] = 0
            ft[:self._ref_phs_ann_class_vars[0]] = 0

            data = np.fft.irfft(ft, axis=0)
            probs = self._get_probs(data, False)

            self._ref_data = np.empty_like(
                self._data_ref_rltzn_srtd, dtype=np.float64)

            for i in range(self._data_ref_n_labels):
                # NOTE: This might not return the reference series.
                self._ref_data[:, i] = self._data_ref_rltzn_srtd[
                    np.argsort(np.argsort(probs[:, i])), i]

        else:
            self._ref_data = self._data_ref_rltzn.copy()

        phs_spec = np.angle(ft)
        mag_spec = np.abs(ft)

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'
        assert np.all(np.isfinite(phs_spec)), 'Invalid values in phs_spec!'
        assert np.all(np.isfinite(mag_spec)), 'Invalid values in mag_spec!'

        self._ref_probs = probs
        self._ref_probs_srtd = np.sort(probs, axis=0)

        self._ref_ft = ft
        self._ref_phs_spec = phs_spec
        self._ref_mag_spec = mag_spec

        if self._sett_obj_cos_sin_dist_flag:
            self._ref_cos_sin_cdfs_dict = self._get_cos_sin_cdfs_dict(
                self._ref_ft)

        self._update_obj_vars('ref')

        self._ref_ft_cumm_corr = self._get_cumm_ft_corr(
            self._ref_ft, self._ref_ft)

        if self._sett_obj_use_obj_dist_flag:
            if self._sett_obj_scorr_flag:
                self._ref_scorr_diffs_cdfs_dict = (
                    self._get_scorr_diffs_cdfs_dict(self._ref_probs))

            if self._sett_obj_asymm_type_1_flag:
                self._ref_asymm_1_diffs_cdfs_dict = (
                    self._get_asymm_1_diffs_cdfs_dict(self._ref_probs))

            if self._sett_obj_asymm_type_2_flag:
                self._ref_asymm_2_diffs_cdfs_dict = (
                    self._get_asymm_2_diffs_cdfs_dict(self._ref_probs))

            if self._sett_obj_ecop_dens_flag:
                self._ref_ecop_dens_diffs_cdfs_dict = (
                    self._get_ecop_dens_diffs_cdfs_dict(self._ref_probs))

            if self._sett_obj_ecop_etpy_flag:
                self._ref_ecop_etpy_diffs_cdfs_dict = (
                    self._get_ecop_etpy_diffs_cdfs_dict(self._ref_probs))

            if self._sett_obj_nth_ord_diffs_flag :
                self._ref_nth_ord_diffs_cdfs_dict = (
                    self._get_nth_ord_diff_cdfs_dict(
                        self._ref_probs, self._sett_obj_nth_ords_vld))

            if self._sett_obj_pcorr_flag:
                self._ref_pcorr_diffs_cdfs_dict = (
                    self._get_pcorr_diffs_cdfs_dict(self._ref_data))

        if self._data_ref_n_labels > 1:
            # NOTE: don't add flags here
            self._ref_mult_asymm_1_diffs_cdfs_dict = (
                self._get_mult_asymm_1_diffs_cdfs_dict(self._ref_probs))

            self._ref_mult_asymm_2_diffs_cdfs_dict = (
                self._get_mult_asymm_2_diffs_cdfs_dict(self._ref_probs))

            self._ref_mult_ecop_dens_diffs_cdfs_dict = (
                self._get_mult_ecop_dens_diffs_cdfs_dict(self._ref_probs))

        self._prep_ref_aux_flag = True
        return

    def _gen_sim_aux_data(self):

        assert self._prep_ref_aux_flag, 'Call _gen_ref_aux_data first!'

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        self._sim_shape = (1 +
            ((self._data_ref_shape[0] *
              self._sett_extnd_len_rel_shp[0]) // 2),
            self._data_ref_n_labels)

#         self._set_phs_ann_cls_vars_sim()

#         ########################################
#         # For testing purposes
#         self._sim_probs = self._ref_probs.copy()
#
#         self._sim_ft = self._ref_ft.copy()
#         self._sim_phs_spec = np.angle(self._ref_ft)
#         self._sim_mag_spec = np.abs(self._ref_ft)
#
#         self._sim_mag_spec_flags = np.ones(self._sim_shape, dtype=bool)
#         self._sim_phs_mod_flags = self._sim_mag_spec_flags.astype(int)
#         #########################################

        if self._sim_phs_mod_flags is None:
            self._sim_phs_mod_flags = np.zeros(self._sim_shape, dtype=int)

            self._sim_phs_mod_flags[+0, :] += 1
            self._sim_phs_mod_flags[-1, :] += 1

            self._sim_n_idxs_all_cts = np.zeros(
                self._sim_shape[0], dtype=np.uint64)

            self._sim_n_idxs_acpt_cts = np.zeros(
                self._sim_shape[0], dtype=np.uint64)

        if self._sett_extnd_len_set_flag:
            ft, mag_spec_flags = self._get_sim_ft_pln(True)

        else:
            ft, mag_spec_flags = self._get_sim_ft_pln()

        # First and last coefficients are not written to anywhere, normally.
        ft[+0] = self._ref_ft[+0].copy()
        ft[-1] = self._ref_ft[-1].copy()

        assert np.all(np.isfinite(ft)), 'Invalid values in ft!'

        data = np.fft.irfft(ft, axis=0)

        assert np.all(np.isfinite(data)), 'Invalid values in data!'

        probs = self._get_probs(data, True)

        self._sim_data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._sim_data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._sim_probs = probs

        self._sim_ft = ft
        self._sim_phs_spec = np.angle(ft)
        self._sim_mag_spec = np.abs(ft)

        self._sim_mag_spec_flags = mag_spec_flags

        self._update_obj_vars('sim')

        if not self._sett_extnd_len_set_flag:
            self._sim_mag_spec_idxs = np.argsort(
                self._sim_mag_spec[1:], axis=0)[::-1, :]

        if self._sett_ann_mag_spec_cdf_idxs_flag:
            # TODO: Have _sim_mag_spec_cdf for current class indices only.
            mag_spec_pdf = (
                self._sim_mag_spec.sum(axis=1) / self._sim_mag_spec.sum())

            mag_spec_cdf = np.concatenate(([0], np.cumsum(mag_spec_pdf)))

            if not extrapolate_flag:
                cdf_ftn = interp1d(
                    mag_spec_cdf,
                    np.arange(mag_spec_cdf.size),
                    bounds_error=True,
                    assume_sorted=True)

            else:
                cdf_ftn = interp1d(
                    mag_spec_cdf,
                    np.arange(mag_spec_cdf.size),
                    bounds_error=False,
                    assume_sorted=True,
                    fill_value='extrapolate',
                    kind='slinear')

#             exct_diffs = cdf_ftn(mag_spec_cdf) - cdf_ftn.y
#
#             assert np.all(np.isclose(exct_diffs, 0.0)), (
#                 'Interpolation function not keeping best estimates!')

            self._sim_mag_spec_cdf = cdf_ftn

        self._prep_sim_aux_flag = True
        return

    def prepare(self):

        '''Generate data required before phase annealing starts'''

        PAS._PhaseAnnealingSettings__verify(self)
        assert self._sett_verify_flag, 'Settings in an unverfied state!'

        self._set_phs_ann_cls_vars_ref()

        self._gen_ref_aux_data()
        assert self._prep_ref_aux_flag, (
            'Apparently, _gen_ref_aux_data did not finish as expected!')

        self._set_phs_ann_cls_vars_sim()

        self._gen_sim_aux_data()
        assert self._prep_sim_aux_flag, (
            'Apparently, _gen_sim_aux_data did not finish as expected!')

        self._prep_prep_flag = True
        return

    def verify(self):

        assert self._prep_prep_flag, 'Call prepare first!'

        if (self._data_ref_n_labels > 1) and self._sett_extnd_len_set_flag:
            raise NotImplementedError(
                'Magnitude annealing with multiple series not implmented!')

        sim_rltzns_out_labs = [
            'ft',
            'mag_spec',
            'phs_spec',
            'probs',
            'scorrs',
            'asymms_1',
            'asymms_2',
            'ecop_dens',
            'ecop_entps',
            'data_ft',
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
            'temps',
            'phs_red_rates',
            'idxs_all',
            'idxs_acpt',
            'acpt_rates_dfrntl',
            'ft_cumm_corr_sim_ref',
            'ft_cumm_corr_sim_sim',
            'phs_ann_class_vars',
            'data',
            'pcorrs',
            'phs_mod_flags',
            'obj_vals_all_indiv',
            'nths',
            'idxs_sclrs',
            ]

        # Order matters for the double for-loops in list-comprehension.
        sim_rltzns_out_labs.extend(
            [f'nth_ord_diffs_{label}_{nth_ord:03d}'
             for label in self._data_ref_labels
             for nth_ord in self._sett_obj_nth_ords_vld])

        sim_rltzns_out_labs.extend(
            [f'scorr_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'asymm_1_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'asymm_2_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'ecop_dens_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'ecop_etpy_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'pcorr_diffs_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        if self._data_ref_n_labels > 1:
            sim_rltzns_out_labs.extend(
                [f'mult_asymm_1_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_1_diffs_cdfs_dict])

            sim_rltzns_out_labs.extend(
                [f'mult_asymm_2_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_2_diffs_cdfs_dict])

#             sim_rltzns_out_labs.extend(
#                 [f'mult_ecop_dens_diffs_{"_".join(comb)}'
#                  for comb in self._ref_mult_ecop_dens_diffs_cdfs_dict])

        # initialize
        self._sim_rltzns_proto_tup = namedtuple(
            'SimRltznData', sim_rltzns_out_labs)

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print(f'Number of classes: {self._ref_phs_ann_n_clss}')

            print(f'Final class width: {self._sett_ann_phs_ann_class_width}')

            print(
                f'Reference annealing class width tuple: '
                f'{self._ref_phs_ann_class_vars}')

            print(
                f'Simulation annealing class width tuple: '
                f'{self._sim_phs_ann_class_vars}')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify

