'''
Created on Dec 27, 2019

@author: Faizan
'''
from collections import namedtuple
from itertools import combinations, product

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rankdata, norm

from ..misc import print_sl, print_el, roll_real_2arrs
from ..cyth import (
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    asymms_exp,
    fill_cumm_dist_from_bivar_emp_dens,
    )

from .settings import PhaseAnnealingSettings as PAS

extrapolate_flag = True
cdf_wts_flag = True


class PhaseAnnealingPrepareTfms:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.
    '''

    def _get_probs(self, data, make_like_ref_flag=False):

        probs_all = np.empty_like(data, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            probs = rankdata(data[:, i]) / (data.shape[0] + 1.0)

            if make_like_ref_flag:
                assert self._ref_probs_srtd is not None

                probs = self._ref_probs_srtd[
                    np.argsort(np.argsort(probs)), i]

            probs_all[:, i] = probs

        return probs_all

    def _get_asymm_1_max(self, scorr):

        a_max = (
            0.5 * (1 - scorr)) * (1 - ((0.5 * (1 - scorr)) ** (1.0 / asymms_exp)))

        return a_max

    def _get_asymm_2_max(self, scorr):

        a_max = (
            0.5 * (1 + scorr)) * (1 - ((0.5 * (1 + scorr)) ** (1.0 / asymms_exp)))

        return a_max

    def _get_etpy_min(self, n_bins):

        # Case for 1-to-1 correlation.
#         dens = 1 / n_bins
#
#         etpy = -np.log(dens)

        # Case for all values in one cell such as that of precipitation.
        etpy = 0.0  # log(1) == 0.
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
            ref_mag[1:-1,:] *
            sim_mag[1:-1,:] *
            np.cos(ref_phs[1:-1,:] - sim_phs[1:-1,:]))

        demr = (
            ((ref_mag[1:-1,:] ** 2).sum(axis=0) ** 0.5) *
            ((sim_mag[1:-1,:] ** 2).sum(axis=0) ** 0.5))

        return np.cumsum(numr, axis=0) / demr

    def _get_sim_ft_pln(self):

        ft = np.zeros(self._sim_shape, dtype=np.complex)

        rands = np.random.random((self._sim_shape[0] - 2, 1))

        rands = 1.0 * (-np.pi + (2 * np.pi * rands))

        rands[~self._ref_phs_sel_idxs] = 0.0

        phs_spec = self._ref_phs_spec[1:-1,:].copy()
        phs_spec += rands  # out of bound phs

        mag_spec = self._ref_mag_spec.copy()

        mag_spec_flags = None

        ft.real[1:-1,:] = mag_spec[1:-1,:] * np.cos(phs_spec)
        ft.imag[1:-1,:] = mag_spec[1:-1,:] * np.sin(phs_spec)

        self._sim_phs_mod_flags[1:-1,:] += 1

        return ft, mag_spec_flags

    def _get_data_ft(self, data, vtype, norm_vals):

        data_ft = np.fft.rfft(data, axis=0)
        data_mag_spec = np.abs(data_ft)[1:-1]

        data_mag_spec = (data_mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (norm_vals is not None):
            data_mag_spec /= norm_vals

        elif (vtype == 'ref') and (norm_vals is None):
            self._ref_data_ft_norm_vals = data_mag_spec[-1,:].copy()

            data_mag_spec /= self._ref_data_ft_norm_vals

        else:
            raise NotImplementedError

        return data_mag_spec

    def _get_probs_ft(self, probs, vtype, norm_vals):

        probs_ft = np.fft.rfft(probs, axis=0)
        probs_mag_spec = np.abs(probs_ft)[1:-1]

        probs_mag_spec = (probs_mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (norm_vals is not None):
            probs_mag_spec /= norm_vals

        elif (vtype == 'ref') and (norm_vals is None):
            self._ref_probs_ft_norm_vals = probs_mag_spec[-1,:].copy()

            probs_mag_spec /= self._ref_probs_ft_norm_vals

        else:
            raise NotImplementedError

        return probs_mag_spec

    def _get_gnrc_ft(self, data, vtype):

        assert data.ndim == 1

        ft = np.fft.rfft(data)
        mag_spec = np.abs(ft)

        mag_spec_sq = mag_spec ** 2

        if ft.real[0] < 0:
            mag_spec_sq[0] *= -1

        mag_spec_cumsum = mag_spec_sq.cumsum()

        if vtype == 'sim':
            norm_val = None
            sclrs = None

        elif vtype == 'ref':
            norm_val = float(mag_spec_cumsum[-1])

            mag_spec_cumsum /= norm_val

            # sclrs lets the first few long amplitudes into account much
            # better. These describe the direction i.e. Asymmetries.
            sclrs = 1.0 / np.arange(1.0, mag_spec_cumsum.size + 1.0)
            sclrs[0] = mag_spec_cumsum.size

        else:
            raise NotImplementedError

        return (mag_spec_cumsum, norm_val, sclrs)


class PhaseAnnealingPrepareCDFS:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.

    NOTE: Add CDF ftns to _trunc_interp_ftns for trunction.
    '''

    def _get_interp_ftn(self, stat_vals_nu, ft_type):

        cdf_vals_nu = rankdata(stat_vals_nu) / (stat_vals_nu.size + 1)

        stat_vals, stat_idxs = np.unique(stat_vals_nu, return_inverse=True)
        cdf_vals, cdf_idxs = np.unique(cdf_vals_nu, return_inverse=True)

        assert np.all(np.isclose(stat_idxs, cdf_idxs)), (
            'Inverse indices not similiar!')

        if extrapolate_flag:
            fill_value = 'extrapolate'

        else:
            fill_value = (
                0 - (20 / stat_vals_nu.size), 1 + (20 / stat_vals_nu.size))

        interp_ftn = interp1d(
            stat_vals,
            cdf_vals,
            bounds_error=False,
            assume_sorted=True,
            fill_value=fill_value,
            kind='slinear')

        assert not hasattr(interp_ftn, 'wts')

        wts = self._get_cdf_wts(
            stat_vals, cdf_vals, stat_vals_nu, cdf_vals_nu, ft_type, stat_idxs)

        interp_ftn.wts = wts

        assert not hasattr(interp_ftn, 'xr')
        assert not hasattr(interp_ftn, 'yr')

        interp_ftn.xr = stat_vals_nu
        interp_ftn.yr = cdf_vals_nu

        # KS test. N and M are same.
        d_nm = 1 / (stat_vals_nu.size ** 0.5)

        d_nm *= (-np.log(0.25)) ** 0.5

        # upper and lower bounds.
        ks_u_bds = cdf_vals_nu - d_nm
        ks_l_bds = cdf_vals_nu + d_nm

        assert not hasattr(interp_ftn, 'ks_u_bds')
        assert not hasattr(interp_ftn, 'ks_l_bds')

        interp_ftn.ks_u_bds = ks_u_bds
        interp_ftn.ks_l_bds = ks_l_bds

        return interp_ftn

    def _get_cdf_wts(
            self,
            diff_vals,
            cdf_vals,
            diff_vals_nu,
            cdf_vals_nu,
            ft_type,
            reconst_idxs):

        '''
        All inputs are assumed to be sorted.

        ft_type can be 1 or 2. 1 for Gaussian-like distributions and 2 for
        exponential-like distributions.
        '''

        if cdf_wts_flag:
            global_min_wt = 1e-6
            min_cdf_diff = 0.5 / cdf_vals_nu.size
            eps = 1e-9

            diff_vals_shft = np.concatenate(
                ([diff_vals[+0] - (0.5 * (diff_vals[+1] - diff_vals[+0]))],
                 diff_vals[:-1] + (0.5 * (diff_vals[1:] - diff_vals[:-1])),
                 [diff_vals[-1] + (0.5 * (diff_vals[-1] - diff_vals[-2]))]))

            cdf_vals_shft = np.concatenate(
                ([cdf_vals[+0] - (0.5 * (cdf_vals[+1] - cdf_vals[+0]))],
                 cdf_vals[:-1] + (0.5 * (cdf_vals[1:] - cdf_vals[:-1])),
                 [cdf_vals[-1] + (0.5 * (cdf_vals[-1] - cdf_vals[-2]))]))

            assert np.all((diff_vals_shft[1:] - diff_vals_shft[:-1]) >= 0), (
                'Differences not positive!')

            assert np.all((cdf_vals_shft[1:] - cdf_vals_shft[:-1]) >= 0), (
                'Differences not positive!')

            cdf_diffs = cdf_vals_shft[1:] - cdf_vals_shft[:-1]
            diff_diffs = diff_vals_shft[1:] - diff_vals_shft[:-1]

            cdf_diffs[cdf_diffs < min_cdf_diff] = min_cdf_diff

            wts = np.abs(diff_diffs / cdf_diffs)

            min_wt = wts[wts > eps].min()
            global_min_wt = max(global_min_wt, min_wt)

#             wts[wts > global_max_wt] = global_max_wt
            wts[wts < global_min_wt] = global_min_wt

#             wts /= np.sum(wts)
            wts *= 0.75
#             wts *= 100

            if diff_vals.size != diff_vals_nu.size:

                fin_wts = wts[reconst_idxs]

                assert fin_wts.size == diff_vals_nu.size

            else:
                fin_wts = wts.copy()

#             import inspect
#             import matplotlib.pyplot as plt
#             plt.plot(diff_vals_nu, fin_wts)
#             plt.title(inspect.stack()[2].function)
#             plt.grid()
#             plt.gca().set_axisbelow(True)
#             plt.show()
#             plt.close()

            assert np.all(fin_wts > 0), 'This was not supposed to happen!'

            assert fin_wts.size == diff_vals_nu.size

        else:
            fin_wts = np.ones_like(diff_vals_nu)

        if self._sett_prt_cdf_calib_set_flag:

            lt = self._sett_prt_cdf_calib_lt
            ut = self._sett_prt_cdf_calib_ut

            lt_n = lt is not None
            ut_n = ut is not None

            assert any([lt_n, ut_n])

            if ft_type == 1:
                pass

            else:
                if lt_n and ut_n:
                    if self._sett_prt_cdf_calib_inside_flag:
                        lt += (1 - ut)

                        ut = None
                        ut_n = ut is not None

                    else:
                        ut -= lt

                        lt = None
                        lt_n = lt is not None

            if lt_n:
                if self._sett_prt_cdf_calib_inside_flag:
                    idxs_lt = cdf_vals_nu >= lt

                else:
                    idxs_lt = cdf_vals_nu <= lt

            if ut_n:
                if self._sett_prt_cdf_calib_inside_flag:
                    idxs_ut = cdf_vals_nu <= ut

                else:
                    idxs_ut = cdf_vals_nu >= ut

            if lt_n and ut_n:
                if self._sett_prt_cdf_calib_inside_flag:
                    idxs = idxs_lt & idxs_ut

                else:
                    idxs = idxs_lt | idxs_ut

            elif lt_n and (not ut_n):
                idxs = idxs_lt.copy()

            elif ut_n and (not lt_n):
                idxs = idxs_ut.copy()

            else:
                raise ValueError('Unknown condition!')

            assert np.any(idxs)

            fin_wts *= idxs.astype(int)

        return fin_wts

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

        if asymms_exp % 2:
            ft_type = 1

        else:
            ft_type = 2

        out_dict = {}
        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            for comb in combs:
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 1 configured for pairs only!')

                diff_vals_nu = np.sort(
                    (probs[:, col_idxs[0]] +
                     probs[:, col_idxs[1]] -
                     1.0) ** asymms_exp)

                out_dict[comb] = self._get_interp_ftn(diff_vals_nu, ft_type)

        return out_dict

    def _get_mult_asymm_2_diffs_cdfs_dict(self, probs):

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        if asymms_exp % 2:
            ft_type = 1

        else:
            ft_type = 2

        out_dict = {}
        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            for comb in combs:
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                diff_vals_nu = np.sort(
                    (probs[:, col_idxs[0]] -
                     probs[:, col_idxs[1]]) ** asymms_exp)

                out_dict[comb] = self._get_interp_ftn(diff_vals_nu, ft_type)

        return out_dict

    def _get_pcorr_diffs_cdfs_dict(self, data):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                data_i, rolled_data_i = roll_real_2arrs(
                    data[:, i], data[:, i], lag)

                diff_vals_nu = np.sort((data_i - rolled_data_i))

                out_dict[(label, lag)] = self._get_interp_ftn(diff_vals_nu, 2)

        return out_dict

    def _get_scorr_diffs_cdfs_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                diff_vals_nu = np.sort(rolled_probs_i * probs_i)

                out_dict[(label, lag)] = self._get_interp_ftn(diff_vals_nu, 2)

        return out_dict

    def _get_asymm_1_diffs_cdfs_dict(self, probs):

        if asymms_exp % 2:
            ft_type = 1

        else:
            ft_type = 2

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                diff_vals_nu = np.sort(
                    (probs_i + rolled_probs_i - 1.0) ** asymms_exp)

                out_dict[(label, lag)] = self._get_interp_ftn(
                    diff_vals_nu, ft_type)

        return out_dict

    def _get_asymm_2_diffs_cdfs_dict(self, probs):

        if asymms_exp % 2:
            ft_type = 1

        else:
            ft_type = 2

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                diff_vals_nu = np.sort(
                    (probs_i - rolled_probs_i) ** asymms_exp)

                out_dict[(label, lag)] = self._get_interp_ftn(
                    diff_vals_nu, ft_type)

        return out_dict

    def _get_asymm_1_diffs_ft_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                out_dict[(label, lag)] = self._get_gnrc_ft(
                    (probs_i + rolled_probs_i - 1.0) ** asymms_exp, 'ref')

        return out_dict

    def _get_asymm_2_diffs_ft_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag)

                out_dict[(label, lag)] = self._get_gnrc_ft(
                    (probs_i - rolled_probs_i) ** asymms_exp, 'ref')

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

                srtd_ecop_dens_nu = np.sort(ecop_dens_arr.ravel())

                out_dict[(label, lag)] = self._get_interp_ftn(
                    srtd_ecop_dens_nu, 2)

                assert not hasattr(out_dict[(label, lag)], 'sclr')

                out_dict[(label, lag)].sclr = (
                    srtd_ecop_dens_nu.size / probs_i.size)

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

                srtd_etpys_arr_nu = np.sort(etpys_arr)

                out_dict[(label, lag)] = self._get_interp_ftn(
                    srtd_etpys_arr_nu, 2)

                assert not hasattr(out_dict[(label, lag)], 'sclr')

                out_dict[(label, lag)].sclr = (
                    srtd_etpys_arr_nu.size / probs_i.size)

        return out_dict

    def _get_cos_sin_ift_dists(self, ft):

        assert ft.ndim == 1

        cosine_ft = np.zeros(ft.size, dtype=complex)
        cosine_ft.real = ft.real
        cosine_ift = np.fft.irfft(cosine_ft)

        sine_ft = np.zeros(ft.size, dtype=complex)
        sine_ft.imag = ft.imag
        sine_ift = np.fft.irfft(sine_ft)

        cosine_ift.sort()
        sine_ift.sort()

        return cosine_ift, sine_ift

    def _get_cos_sin_cdfs_dict(self, ft):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            cos_vals_nu, sin_vals_nu = self._get_cos_sin_ift_dists(ft[:, i])

            out_dict[(label, 'cos')] = self._get_interp_ftn(cos_vals_nu, 1)
            out_dict[(label, 'sin')] = self._get_interp_ftn(sin_vals_nu, 1)

            assert not hasattr(out_dict[(label, 'cos')], 'sclr')
            assert not hasattr(out_dict[(label, 'sin')], 'sclr')

            out_dict[(label, 'cos')].sclr = 2
            out_dict[(label, 'sin')].sclr = out_dict[(label, 'cos')].sclr

        return out_dict

    def _get_srtd_nth_diffs_arrs(self, vals, nth_ords, max_nth_ords=None):

        assert self._sett_obj_nth_ords is not None, 'nth_ords not defined!'

        srtd_nth_ord_diffs_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            if max_nth_ords is not None:
                max_nth_ord = max_nth_ords[label]

            else:
                max_nth_ord = nth_ords.max()

            diffs = None
            for nth_ord in np.arange(1, max_nth_ord + 1, dtype=np.int64):

                if diffs is None:
                    diffs = vals[:-1, i] - vals[1:, i]

                else:
                    diffs = diffs[:-1] - diffs[1:]

                if nth_ord not in nth_ords:
                    continue

                srtd_nth_ord_diffs_dict[(label, nth_ord)] = np.sort(diffs)

        return srtd_nth_ord_diffs_dict

    def _get_nth_ord_diffs_cdfs_dict(self, vals, nth_ords):

        diffs_dict = self._get_srtd_nth_diffs_arrs(vals, nth_ords)

        nth_ords_cdfs_dict = {}
        for lab_nth_ord, diff_vals_nu in diffs_dict.items():

            nth_ords_cdfs_dict[lab_nth_ord] = self._get_interp_ftn(
                diff_vals_nu, 1)

        return nth_ords_cdfs_dict

    def _get_nth_diff_ft_arrs(self, vals, nth_ords, vtype, max_nth_ords=None):

        nth_ord_diffs_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            if max_nth_ords is not None:
                max_nth_ord = max_nth_ords[label]

            else:
                max_nth_ord = nth_ords.max()

            diffs = None
            for nth_ord in np.arange(1, max_nth_ord + 1, dtype=np.int64):

                if diffs is None:
                    diffs = vals[:-1, i] - vals[1:, i]

                else:
                    diffs = diffs[:-1] - diffs[1:]

                if nth_ord not in nth_ords:
                    continue

                # NOTE: Much better with a cube.
                # But overflow errors in case values are too big.
                # This only takes place when optimization takes place,
                # the outputs that are saved and plotted are for the original
                # nth order differences.
                if (not self._alg_done_opt_flag) and (nth_ord == 1):
                    diffs **= 3

                nth_ord_diffs_dict[(label, nth_ord)] = self._get_gnrc_ft(
                    diffs, vtype)

                if vtype == 'sim':
                    nth_ord_diffs_dict[(label, nth_ord)] = (
                        nth_ord_diffs_dict[(label, nth_ord)][0])

                    nth_ord_diffs_dict[(label, nth_ord)] /= (
                        self._ref_nth_ord_diffs_ft_dict[(label, nth_ord)][1])

                elif vtype == 'ref':
                    pass

                else:
                    raise NotImplementedError

        return nth_ord_diffs_dict

    def _get_nth_ord_diffs_ft_dict(self, vals, nth_ords):

        nth_ords_ft_dict = self._get_nth_diff_ft_arrs(vals, nth_ords, 'ref')

        return nth_ords_ft_dict


class PhaseAnnealingPrepare(
        PAS,
        PhaseAnnealingPrepareTfms,
        PhaseAnnealingPrepareCDFS):

    '''Prepare derived variables required by phase annealing here'''

    def __init__(self, verbose=True):

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
        self._ref_probs_srtd = None
        self._ref_data = None
        self._ref_pcorrs = None
        self._ref_nths = None
        self._ref_data_ft = None
        self._ref_data_ft_norm_vals = None
        self._ref_data_tfm = None
        self._ref_phs_sel_idxs = None
        self._ref_phs_idxs = None
        self._ref_probs_ft = None
        self._ref_probs_ft_norm_vals = None

        self._data_tfm_type = 'norm'
        self._data_tfm_types = (
            'log_data', 'probs', 'data', 'probs_sqrt', 'norm')

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

        self._ref_scorr_qq_dict = None
        self._ref_asymm_1_qq_dict = None
        self._ref_asymm_2_qq_dict = None
        self._ref_ecop_dens_qq_dict = None
        self._ref_ecop_etpy_qq_dict = None
        self._ref_nth_ord_qq_dict = None
        self._ref_pcorr_qq_dict = None

        self._ref_mult_asymm_1_qq_dict = None
        self._ref_mult_asymm_2_qq_dict = None

        self._ref_asymm_1_diffs_ft_dict = None
        self._ref_asymm_2_diffs_ft_dict = None
        self._ref_nth_ord_diffs_ft_dict = None

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

        self._sim_shape = None
        self._sim_mag_spec_cdf = None
        self._sim_data = None
        self._sim_pcorrs = None
        self._sim_nths = None
        self._sim_ft_best = None
        self._sim_data_ft = None
        self._sim_probs_ft = None

        # To keep track of modified phases
        self._sim_phs_mod_flags = None
        self._sim_n_idxs_all_cts = None
        self._sim_n_idxs_acpt_cts = None

        # An array. False for phase changes, True for coeff changes
        self._sim_mag_spec_flags = None

        # Objective function variables.
        self._sim_scorr_diffs = None
        self._sim_asymm_1_diffs = None
        self._sim_asymm_2_diffs = None
        self._sim_ecop_dens_diffs = None
        self._sim_ecop_etpy_diffs = None
        self._sim_nth_ord_diffs = None
        self._sim_pcorr_diffs = None

        self._sim_mult_asymms_1_diffs = None
        self._sim_mult_asymms_2_diffs = None

        self._sim_asymm_1_diffs_ft = None
        self._sim_asymm_2_diffs_ft = None
        self._sim_nth_ord_diffs_ft = None

        # QQ probs
        self._sim_scorr_qq_dict = None
        self._sim_asymm_1_qq_dict = None
        self._sim_asymm_2_qq_dict = None
        self._sim_ecop_dens_qq_dict = None
        self._sim_ecop_etpy_qq_dict = None
        self._sim_nth_ords_qq_dict = None
        self._sim_pcorr_qq_dict = None

        self._sim_mult_asymm_1_qq_dict = None
        self._sim_mult_asymm_2_qq_dict = None
        self._sim_mult_ecop_dens_qq_dict = None

        # Misc.
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

    def _set_sel_phs_idxs(self):

        periods = self._ref_probs.shape[0] / (
            np.arange(1, self._ref_ft.shape[0] - 1))

        self._ref_phs_sel_idxs = np.ones(
            self._ref_ft.shape[0] - 2, dtype=bool)

        if self._sett_sel_phs_min_prd is not None:
            assert periods.min() <= self._sett_sel_phs_min_prd, (
                'Minimum period does not exist in data!')

            assert periods.max() > self._sett_sel_phs_min_prd, (
                'Data maximum period greater than or equal to min_period!')

            self._ref_phs_sel_idxs[
                periods < self._sett_sel_phs_min_prd] = False

        if self._sett_sel_phs_max_prd is not None:
            assert periods.max() >= self._sett_sel_phs_max_prd, (
                'Maximum period does not exist in data!')

            self._ref_phs_sel_idxs[
                periods > self._sett_sel_phs_max_prd] = False

        assert self._ref_phs_sel_idxs.sum(), (
            'Incorrect min_period or max_period, '
            'not phases selected for phsann!')
        return

    @PAS._timer_wrap
    def _update_obj_vars(self, vtype):

        '''Required variables e.g. self._XXX_probs should have been
        defined/updated before entering.
        '''

        assert any([
            self._sett_obj_use_obj_lump_flag,
            self._sett_obj_use_obj_dist_flag])

        # Will raise an error if the algorithm class is not there.
        cont_flag_01_prt = (
            (not self._alg_wts_lag_nth_search_flag) and
            (self._sett_wts_lags_nths_set_flag) and
            (not self._alg_done_opt_flag))

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

            if self._sett_obj_use_obj_lump_flag:
                scorrs = np.full(
                    (self._data_ref_n_labels, lag_steps.size),
                    np.nan)

            else:
                scorrs = None

            if (self._sett_obj_use_obj_dist_flag and
                self._sett_obj_scorr_flag and
                (vtype == 'sim')):

                scorr_diffs = {}

                if cont_flag_01_prt and (self._alg_wts_lag_scorr is not None):
                    scorr_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_scorr[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    scorr_diff_conts = {}

            else:
                scorr_diffs = None

        else:
            scorrs = None
            scorr_diffs = None

        if self._sett_obj_pcorr_flag:
            if self._sett_obj_use_obj_lump_flag:
                pcorrs = np.full(
                    (self._data_ref_n_labels, lag_steps.size),
                    np.nan)

            else:
                pcorrs = None

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                pcorr_diffs = {}

                if cont_flag_01_prt and (self._alg_wts_lag_pcorr is not None):
                    pcorr_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_pcorr[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    pcorr_diff_conts = {}

            else:
                pcorr_diffs = None

        else:
            pcorrs = None
            pcorr_diffs = None

        if self._sett_obj_asymm_type_1_flag:
            if self._sett_obj_use_obj_lump_flag:
                asymms_1 = np.full(
                    (self._data_ref_n_labels, lag_steps.size),
                    np.nan)

            else:
                asymms_1 = None

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                asymm_1_diffs = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_asymm_1 is not None)):

                    asymm_1_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_asymm_1[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    asymm_1_diff_conts = {}

            else:
                asymm_1_diffs = None

        else:
            asymms_1 = None
            asymm_1_diffs = None

        if self._sett_obj_asymm_type_2_flag:
            if self._sett_obj_use_obj_lump_flag:
                asymms_2 = np.full(
                    (self._data_ref_n_labels, lag_steps.size),
                    np.nan)

            else:
                asymms_2 = None

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                asymm_2_diffs = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_asymm_2 is not None)):

                    asymm_2_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_asymm_2[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    asymm_2_diff_conts = {}

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

            # first row and first col are temps. They are always zeros.
            ecop_cumm_dens_arrs = np.zeros(
                (self._data_ref_n_labels,
                 lag_steps.size,
                 self._sett_obj_ecop_dens_bins + 1,
                 self._sett_obj_ecop_dens_bins + 1),
                dtype=np.float64)

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                ecop_dens_diffs = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_ecop_dens is not None)):

                    ecop_dens_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_ecop_dens[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    ecop_dens_diff_conts = {}

            else:
                ecop_dens_diffs = None

        else:
            ecop_dens_arrs = None
            ecop_cumm_dens_arrs = None
            ecop_dens_diffs = None

        if self._sett_obj_ecop_etpy_flag:
            if self._sett_obj_use_obj_lump_flag:
                ecop_etpy_arrs = np.full(
                    (self._data_ref_n_labels, lag_steps.size,),
                    np.nan,
                    dtype=np.float64)

                etpy_min = self._get_etpy_min(self._sett_obj_ecop_dens_bins)
                etpy_max = self._get_etpy_max(self._sett_obj_ecop_dens_bins)

            else:
                ecop_etpy_arrs = etpy_min = etpy_max = None

            if self._sett_obj_use_obj_dist_flag and (vtype == 'sim'):
                ecop_etpy_diffs = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_ecop_etpy is not None)):

                    ecop_etpy_diff_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_ecop_etpy[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    ecop_etpy_diff_conts = {}

            else:
                ecop_etpy_diffs = None

        else:
            ecop_etpy_arrs = etpy_min = etpy_max = None
            ecop_etpy_diffs = None

        if self._sett_obj_nth_ord_diffs_flag:
            if self._sett_obj_use_obj_lump_flag:
                nths = np.full((
                    self._data_ref_n_labels, nth_ords.size), np.nan)

            else:
                nths = None

            if ((cont_flag_01_prt) and
                (self._alg_wts_nth_order is not None)):

                nth_ord_diff_conts = {}

                for label in self._data_ref_labels:
                    max_nth_ord = np.nan
                    for nth_ord in nth_ords[::-1]:
                        if self._alg_wts_nth_order[(label, nth_ord)]:
                            max_nth_ord = nth_ord
                            break

                    nth_ord_diff_conts[label] = max_nth_ord

            else:
                nth_ord_diff_conts = None

            nth_ord_diffs = self._get_srtd_nth_diffs_arrs(
                data, nth_ords, nth_ord_diff_conts)

            if nths is not None:
                for j, label in enumerate(self._data_ref_labels):
                    for i, nth_ord in enumerate(nth_ords):
                        nths[j, i] = np.sum(nth_ord_diffs[(label, nth_ord)])

            if not self._sett_obj_use_obj_dist_flag:
                nth_ord_diffs = None

        else:
            nth_ord_diffs = None
            nths = None

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

        if self._sett_obj_match_probs_ft_flag:
            if vtype == 'sim':
                probs_ft_norm_vals = self._ref_probs_ft_norm_vals

            else:
                probs_ft_norm_vals = None

            probs_ft = self._get_probs_ft(probs, vtype, probs_ft_norm_vals)

        else:
            probs_ft = None

        if self._sett_obj_asymm_type_1_ft_flag:
            if vtype == 'sim':
                asymm_1_diffs_ft = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_asymm_1_ft is not None)):

                    asymm_1_diff_ft_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_asymm_1_ft[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    asymm_1_diff_ft_conts = {}

            else:
                asymm_1_diffs_ft = None

        else:
            asymm_1_diffs_ft = None

        if self._sett_obj_asymm_type_2_ft_flag:
            if vtype == 'sim':
                asymm_2_diffs_ft = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_asymm_2_ft is not None)):

                    asymm_2_diff_ft_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_asymm_2_ft[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    asymm_2_diff_ft_conts = {}

            else:
                asymm_2_diffs_ft = None

        else:
            asymm_2_diffs_ft = None

        if self._sett_obj_nth_ord_diffs_ft_flag and (vtype == 'sim'):
            if ((cont_flag_01_prt) and
                (self._alg_wts_nth_order_ft is not None)):

                nth_ord_diff_ft_conts = {}

                for label in self._data_ref_labels:
                    max_nth_ord = np.nan
                    for nth_ord in nth_ords[::-1]:
                        if self._alg_wts_nth_order_ft[(label, nth_ord)]:
                            max_nth_ord = nth_ord
                            break

                    nth_ord_diff_ft_conts[label] = max_nth_ord

            else:
                nth_ord_diff_ft_conts = None

            nth_ord_diffs_ft = self._get_nth_diff_ft_arrs(
                data, nth_ords, vtype, nth_ord_diff_ft_conts)

        else:
            nth_ord_diffs_ft = None

        c_scorrs = scorrs is not None
        c_scorr_diffs = scorr_diffs is not None
        c_asymms_1 = asymms_1 is not None
        c_asymm_1_diffs = asymm_1_diffs is not None
        c_asymms_2 = asymms_2 is not None
        c_asymms_2_diffs = asymm_2_diffs is not None
        c_ecop_dens_arrs = ecop_dens_arrs is not None
        c_ecop_dens_diffs = ecop_dens_diffs is not None
        c_ecop_etpy_arrs = ecop_etpy_arrs is not None
        c_ecop_etpy_diffs = ecop_etpy_diffs is not None
        c_pcorrs = pcorrs is not None
        c_pcorr_diffs = pcorr_diffs is not None
        c_asymm_1_diffs_ft = asymm_1_diffs_ft is not None
        c_asymm_2_diffs_ft = asymm_2_diffs_ft is not None

        ca = any([
            c_scorrs,
            c_scorr_diffs,
            c_asymms_1,
            c_asymm_1_diffs,
            c_asymms_2,
            c_asymms_2_diffs,
            c_ecop_dens_arrs,
            c_ecop_dens_diffs,
            c_ecop_etpy_arrs,
            c_ecop_etpy_diffs,
            c_asymm_1_diffs_ft,
            c_asymm_2_diffs_ft])

        cb = any([
            c_pcorrs,
            c_pcorr_diffs])

        loop_prod = product(
            enumerate(self._data_ref_labels), enumerate(lag_steps))

        for ((j, label), (i, lag)) in loop_prod:

            if (not ca) and (not cb):
                break

            if ca:
                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, j], probs[:, j], lag)

                if c_scorrs:
                    scorrs[j, i] = np.corrcoef(probs_i, rolled_probs_i)[0, 1]

                if c_scorr_diffs and scorr_diff_conts.get((label, lag), True):
                    scorr_diffs[(label, lag)] = np.sort(
                        rolled_probs_i * probs_i)

                if c_asymms_1:
                    asymms_1[j, i] = get_asymm_1_sample(
                        probs_i, rolled_probs_i)

                    asymms_1[j, i] /= self._get_asymm_1_max(scorrs[j, i])

                if (c_asymm_1_diffs and
                    asymm_1_diff_conts.get((label, lag), True)):

                    asymm_1_diffs[(label, lag)] = np.sort(
                        (probs_i + rolled_probs_i - 1.0) ** asymms_exp)

                if c_asymms_2:
                    asymms_2[j, i] = get_asymm_2_sample(
                        probs_i, rolled_probs_i)

                    asymms_2[j, i] /= self._get_asymm_2_max(scorrs[j, i])

                if (c_asymms_2_diffs and
                    asymm_2_diff_conts.get((label, lag), True)):

                    asymm_2_diffs[(label, lag)] = np.sort(
                        (probs_i - rolled_probs_i) ** asymms_exp)

                if c_ecop_dens_arrs:
                    fill_bi_var_cop_dens(
                        probs_i, rolled_probs_i, ecop_dens_arrs[j, i,:,:])

                    fill_cumm_dist_from_bivar_emp_dens(
                        ecop_dens_arrs[j, i,:,:],
                        ecop_cumm_dens_arrs[j, i,:,:])

                if (c_ecop_dens_diffs and
                    ecop_dens_diff_conts.get((label, lag), True)):

                    ecop_dens_diffs[(label, lag)] = np.sort(
                        ecop_dens_arrs[j, i,:,:].ravel())

                if ((c_ecop_etpy_arrs) or
                    (c_ecop_etpy_diffs and
                     ecop_etpy_diff_conts.get((label, lag), True))):

                    non_zero_idxs = ecop_dens_arrs[j, i,:,:] > 0

                    dens = ecop_dens_arrs[j, i][non_zero_idxs]

                    etpy_arr = -(dens * np.log(dens))

                if c_ecop_etpy_arrs:
                    etpy = etpy_arr.sum()

                    etpy = (etpy - etpy_min) / (etpy_max - etpy_min)

                    assert 0 <= etpy <= 1, f'etpy {etpy:0.3f} out of bounds!'

                    ecop_etpy_arrs[j, i] = etpy

                if (c_ecop_etpy_diffs and
                    ecop_etpy_diff_conts.get((label, lag), True)):

                    etpy_diffs = np.zeros(
                        self._sett_obj_ecop_dens_bins ** 2)

                    etpy_diffs[non_zero_idxs.ravel()] = etpy_arr

                    ecop_etpy_diffs[(label, lag)] = np.sort(etpy_diffs)

                if (c_asymm_1_diffs_ft and
                    asymm_1_diff_ft_conts.get((label, lag), True)):

                    asymm_1_diffs_ft[(label, lag)] = self._get_gnrc_ft(
                        (probs_i + rolled_probs_i - 1.0) ** asymms_exp,
                        'sim')[0]

                    asymm_1_diffs_ft[(label, lag)] /= (
                        self._ref_asymm_1_diffs_ft_dict[(label, lag)])[1]

                if (c_asymm_2_diffs_ft and
                    asymm_2_diff_ft_conts.get((label, lag), True)):

                    asymm_2_diffs_ft[(label, lag)] = self._get_gnrc_ft(
                        (probs_i - rolled_probs_i) ** asymms_exp, 'sim')[0]

                    asymm_2_diffs_ft[(label, lag)] /= (
                        self._ref_asymm_2_diffs_ft_dict[(label, lag)])[1]

            if cb:
                if ((c_pcorrs) or
                    (c_pcorr_diffs and
                     pcorr_diff_conts.get((label, lag), True))):

                    data_i, rolled_data_i = roll_real_2arrs(
                        data[:, j], data[:, j], lag)

                if c_pcorrs:
                    pcorrs[j, i] = np.corrcoef(data_i, rolled_data_i)[0, 1]

                if c_pcorr_diffs and pcorr_diff_conts.get((label, lag), True):
                    pcorr_diffs[(label, lag)] = np.sort(
                        (data_i - rolled_data_i))

        if mult_asymm_1_diffs is not None:
            for comb in self._ref_mult_asymm_1_diffs_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 1 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] + probs[:, col_idxs[1]] - 1.0
                        ) ** asymms_exp)

                mult_asymm_1_diffs[comb] = diff_vals

        if mult_asymm_2_diffs is not None:
            for comb in self._ref_mult_asymm_2_diffs_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                diff_vals = np.sort(
                    (probs[:, col_idxs[0]] - probs[:, col_idxs[1]]
                        ) ** asymms_exp)

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

#         if scorrs is not None:
#             assert np.all(np.isfinite(scorrs)), 'Invalid values in scorrs!'
#
#             assert np.all((scorrs >= -1.0) & (scorrs <= +1.0)), (
#                 'scorrs out of range!')
#
#
#         if asymms_1 is not None:
#             assert np.all(np.isfinite(asymms_1)), 'Invalid values in asymms_1!'
#
#             assert np.all((asymms_1 >= -1.0) & (asymms_1 <= +1.0)), (
#                 'asymms_1 out of range!')
#
#         if asymms_2 is not None:
#             assert np.all(np.isfinite(asymms_2)), 'Invalid values in asymms_2!'
#
#             assert np.all((asymms_2 >= -1.0) & (asymms_2 <= +1.0)), (
#                 'asymms_2 out of range!')
#
#         if ecop_dens_arrs is not None:
#             assert np.all(np.isfinite(ecop_dens_arrs)), (
#                 'Invalid values in ecop_dens_arrs!')
#
#         if ecop_etpy_arrs is not None:
#             assert np.all(np.isfinite(ecop_etpy_arrs)), (
#                 'Invalid values in ecop_etpy_arrs!')
#
#             assert np.all(ecop_etpy_arrs >= 0), (
#                 'ecop_etpy_arrs values out of range!')
#
#             assert np.all(ecop_etpy_arrs <= 1), (
#                 'ecop_etpy_arrs values out of range!')
#
#         if nth_ord_diffs is not None:
#             for lab_nth_ord in nth_ord_diffs:
#                 assert np.all(np.isfinite(nth_ord_diffs[lab_nth_ord])), (
#                     'Invalid values in nth_ord_diffs!')
#
#         if pcorrs is not None:
#             assert np.all(np.isfinite(pcorrs)), 'Invalid values in pcorrs!'

        if not self._sett_obj_scorr_flag:
            scorrs = None

        if vtype == 'ref':
            self._ref_scorrs = scorrs
            self._ref_asymms_1 = asymms_1
            self._ref_asymms_2 = asymms_2
            self._ref_ecop_dens = ecop_cumm_dens_arrs  # ecop_dens_arrs
            self._ref_ecop_etpy = ecop_etpy_arrs
            self._ref_pcorrs = pcorrs
            self._ref_nths = nths
            self._ref_data_ft = data_ft
            self._ref_probs_ft = probs_ft

        elif vtype == 'sim':
            # NOTE: Update the snapshot method in Algorithm accordingly
            self._sim_scorrs = scorrs
            self._sim_asymms_1 = asymms_1
            self._sim_asymms_2 = asymms_2
            self._sim_ecop_dens = ecop_cumm_dens_arrs  # ecop_dens_arrs
            self._sim_ecop_etpy = ecop_etpy_arrs
            self._sim_pcorrs = pcorrs
            self._sim_nths = nths
            self._sim_data_ft = data_ft
            self._sim_probs_ft = probs_ft

            self._sim_scorr_diffs = scorr_diffs
            self._sim_asymm_1_diffs = asymm_1_diffs
            self._sim_asymm_2_diffs = asymm_2_diffs
            self._sim_ecop_dens_diffs = ecop_dens_diffs
            self._sim_ecop_etpy_diffs = ecop_etpy_diffs
            self._sim_nth_ord_diffs = nth_ord_diffs
            self._sim_pcorr_diffs = pcorr_diffs

            self._sim_asymm_1_diffs_ft = asymm_1_diffs_ft
            self._sim_asymm_2_diffs_ft = asymm_2_diffs_ft
            self._sim_nth_ord_diffs_ft = nth_ord_diffs_ft

            self._sim_mult_asymms_1_diffs = mult_asymm_1_diffs
            self._sim_mult_asymms_2_diffs = mult_asymm_2_diffs
            self._sim_mult_ecops_dens_diffs = mult_ecop_dens_diffs

        else:
            raise ValueError(f'Unknown vtype in _update_obj_vars: {vtype}!')

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
                f'_ref_data_tfm_type can only be from: '
                f'{self._data_tfm_types}!')

        assert np.all(np.isfinite(data_tfm)), 'Invalid values in data_tfm!'

        return data_tfm

    def _gen_ref_aux_data(self):

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implementation for 2D only!')

        probs = self._get_probs(self._data_ref_rltzn, False)

        self._ref_data_tfm = self._get_data_tfm(self._data_ref_rltzn, probs)

        ft = np.fft.rfft(self._ref_data_tfm, axis=0)

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

        self._set_sel_phs_idxs()

        self._ref_phs_idxs = np.arange(
            1, self._ref_ft.shape[0] - 1)[self._ref_phs_sel_idxs]

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

            if self._sett_obj_nth_ord_diffs_flag:
                self._ref_nth_ord_diffs_cdfs_dict = (
                    self._get_nth_ord_diffs_cdfs_dict(
                        self._ref_data, self._sett_obj_nth_ords_vld))

            if self._sett_obj_pcorr_flag:
                self._ref_pcorr_diffs_cdfs_dict = (
                    self._get_pcorr_diffs_cdfs_dict(self._ref_data))

            if self._sett_obj_asymm_type_1_ft_flag:
                self._ref_asymm_1_diffs_ft_dict = (
                    self._get_asymm_1_diffs_ft_dict(self._ref_probs))

            if self._sett_obj_asymm_type_2_ft_flag:
                self._ref_asymm_2_diffs_ft_dict = (
                    self._get_asymm_2_diffs_ft_dict(self._ref_probs))

            if self._sett_obj_nth_ord_diffs_ft_flag:
                self._ref_nth_ord_diffs_ft_dict = (
                    self._get_nth_ord_diffs_ft_dict(
                        self._ref_data, self._sett_obj_nth_ords_vld))

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

        self._sim_shape = (1 + (self._data_ref_shape[0] // 2),
            self._data_ref_n_labels)

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

            self._sim_phs_mod_flags[+0,:] += 1
            self._sim_phs_mod_flags[-1,:] += 1

            self._sim_n_idxs_all_cts = np.zeros(
                self._sim_shape[0], dtype=np.uint64)

            self._sim_n_idxs_acpt_cts = np.zeros(
                self._sim_shape[0], dtype=np.uint64)

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
            self._sim_data[:, i] = self._data_ref_rltzn_srtd.copy()[
                np.argsort(np.argsort(probs[:, i])), i]

        self._sim_probs = probs

        self._sim_ft = ft
        self._sim_phs_spec = np.angle(ft)
        self._sim_mag_spec = np.abs(ft)

        self._sim_mag_spec_flags = mag_spec_flags

        self._update_obj_vars('sim')

        self._sim_mag_spec_idxs = np.argsort(
            self._sim_mag_spec[1:], axis=0)[::-1,:]

        if self._sett_ann_mag_spec_cdf_idxs_flag:
            mag_spec = self._sim_mag_spec.copy()

            mag_spec = mag_spec[self._ref_phs_idxs]

            mag_spec_pdf = mag_spec.sum(axis=1) / mag_spec.sum()

            assert np.all(mag_spec_pdf > 0), (
                'Phases with zero magnitude not allowed!')

            mag_spec_pdf = 1 / mag_spec_pdf

            mag_spec_pdf /= mag_spec_pdf.sum()

            assert np.all(np.isfinite(mag_spec_pdf)), (
                'Invalid values in mag_spec_pdf!')

            mag_spec_cdf = mag_spec_pdf.copy()

            self._sim_mag_spec_cdf = mag_spec_cdf

        self._prep_sim_aux_flag = True
        return

    def prepare(self):

        '''Generate data required before phase annealing starts'''

        PAS._PhaseAnnealingSettings__verify(self)
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
            'probs_ft',
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
            'data',
            'pcorrs',
            'phs_mod_flags',
            'obj_vals_all_indiv',
            'nths',
            'idxs_sclrs',
            'tmr_call_times',
            'tmr_n_calls',
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

        sim_rltzns_out_labs.extend(
            [f'asymm_1_diffs_ft_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'asymm_2_diffs_ft_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'nth_ord_diffs_ft_{label}_{nth_ord:03d}'
             for label in self._data_ref_labels
             for nth_ord in self._sett_obj_nth_ords_vld])

        if self._data_ref_n_labels > 1:
            sim_rltzns_out_labs.extend(
                [f'mult_asymm_1_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_1_diffs_cdfs_dict])

            sim_rltzns_out_labs.extend(
                [f'mult_asymm_2_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_2_diffs_cdfs_dict])

        # Same for QQ probs
        sim_rltzns_out_labs.extend(
            [f'scorr_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'asymm_1_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'asymm_2_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'ecop_dens_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'ecop_etpy_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        sim_rltzns_out_labs.extend(
            [f'nth_ord_qq_{label}_{nth_ord:03d}'
             for label in self._data_ref_labels
             for nth_ord in self._sett_obj_nth_ords_vld])

        sim_rltzns_out_labs.extend(
            [f'pcorr_qq_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        if self._data_ref_n_labels > 1:
            sim_rltzns_out_labs.extend(
                [f'mult_asymm_1_qq_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_1_diffs_cdfs_dict])

            sim_rltzns_out_labs.extend(
                [f'mult_asymm_2_qq_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_2_diffs_cdfs_dict])

        # initialize
        self._sim_rltzns_proto_tup = namedtuple(
            'SimRltznData', sim_rltzns_out_labs)

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify

