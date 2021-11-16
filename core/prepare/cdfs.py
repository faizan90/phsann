'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from itertools import combinations

import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import rankdata

from fcopulas import (
    fill_bi_var_cop_dens,
    asymms_exp,
    )

from ...misc import (
    roll_real_2arrs,
    get_local_entropy_ts_cy,
    )

# These flags are only used here, so not putting them up as cnsts in prepare.
extrapolate_flag = True
cdf_wts_flag = True
empirical_wts_flag = False  # Cmpted from empr dens ftn or Anderson-Darling.


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

        # Upper and lower bounds.
        ks_u_bds = cdf_vals_nu - d_nm
        ks_l_bds = cdf_vals_nu + d_nm

        assert not hasattr(interp_ftn, 'ks_u_bds')
        assert not hasattr(interp_ftn, 'ks_l_bds')

        interp_ftn.ks_u_bds = ks_u_bds
        interp_ftn.ks_l_bds = ks_l_bds

        vals_per_bin = max(2, int(
            self._sett_obj_ratio_per_dens_bin * stat_vals_nu.size))

        if self._sett_obj_use_dens_ftn_flag:
            assert stat_vals_nu.size > 2, (
                'Too few values for empirical density function computation!')

            assert vals_per_bin > 1, (
                    f'ratio_per_dens_bin ({vals_per_bin}) is too '
                    f'small for the given number of data points and '
                    f'lag steps / nth orders!')

        bins = []
        for i in range(0, stat_vals_nu.size, vals_per_bin):
            bins.append(stat_vals_nu[i])

        if bins[-1] != stat_vals_nu[-1]:
            bins.append(stat_vals_nu[-1])

        hist = np.histogram(
            stat_vals_nu,
            bins=bins,
            range=(bins[0], bins[-1]),
            )[0] / stat_vals_nu.size

        assert not hasattr(interp_ftn, 'bins')
        assert not hasattr(interp_ftn, 'hist')

        interp_ftn.bins = bins
        interp_ftn.hist = hist

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
        exponential-like distributions. Only used when the flag
        self._sett_prt_cdf_calib_set_flag is set to True.
        '''

        if cdf_wts_flag and empirical_wts_flag:
            # Weights using the empirical density function.
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

        elif cdf_wts_flag and (not empirical_wts_flag):
            # Weights based on Anderson-Darling distribution fitting test.
            if ft_type == 1:
                wts = 1 / (cdf_vals * (1 - cdf_vals))
                wts /= wts.sum()

            elif ft_type == 2:
                wts = 1 / (1 - cdf_vals)
                wts /= wts.sum()

            else:
                raise NotImplementedError

            if diff_vals.size != diff_vals_nu.size:

                fin_wts = wts[reconst_idxs]

                assert fin_wts.size == diff_vals_nu.size

            else:
                fin_wts = wts.copy()

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

            elif ft_type == 2:
                if lt_n and ut_n:
                    if self._sett_prt_cdf_calib_inside_flag:
                        lt += (1 - ut)

                        ut = None
                        ut_n = ut is not None

                    else:
                        ut -= lt

                        lt = None
                        lt_n = lt is not None

            else:
                raise NotImplementedError

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

    def _get_mult_asymm_1_cmpos_ft(self, probs, vtype):

        return self._get_gnrc_mult_ft(probs, vtype, 'asymm1')

    def _get_mult_asymm_2_cmpos_ft(self, probs, vtype):

        return self._get_gnrc_mult_ft(probs, vtype, 'asymm2')

    def _get_mult_etpy_cmpos_ft(self, probs, vtype):

        return self._get_gnrc_mult_ft(probs, vtype, 'etpy')

#     def _get_mult_scorr_cmpos_ft(self, probs, vtype):
#
#         return self._get_gnrc_mult_ft(probs, vtype, 'corr')

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

                srtd_ecop_dens_nu = np.sort(ecop_dens_arr.ravel())

                out_dict[comb] = self._get_interp_ftn(srtd_ecop_dens_nu, 2)

                assert not hasattr(out_dict[comb], 'sclr')

                out_dict[comb].sclr = srtd_ecop_dens_nu.size / probs.shape[0]

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
                    data[:, i], data[:, i], lag, False)

                diff_vals_nu = np.sort((data_i - rolled_data_i))

                out_dict[(label, lag)] = self._get_interp_ftn(diff_vals_nu, 2)

        return out_dict

    def _get_scorr_diffs_cdfs_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag, True)

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
                    probs[:, i], probs[:, i], lag, True)

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
                    probs[:, i], probs[:, i], lag, True)

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
                    probs[:, i], probs[:, i], lag, True)

                out_dict[(label, lag)] = self._get_gnrc_ft(
                    (probs_i + rolled_probs_i - 1.0) ** asymms_exp, 'ref')

        return out_dict

    def _get_asymm_2_diffs_ft_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag, True)

#                 pdf_ts = get_pdf_ts(
#                     (probs_i - rolled_probs_i) ** asymms_exp, 200)

                out_dict[(label, lag)] = self._get_gnrc_ft(
                    (probs_i - rolled_probs_i) ** asymms_exp, 'ref')

        return out_dict

    def _get_etpy_ft_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag, True)

                etpy_ts = get_local_entropy_ts_cy(
                    probs_i, rolled_probs_i, self._sett_obj_ecop_dens_bins)

                assert np.all(np.isfinite(etpy_ts))

                out_dict[(label, lag)] = self._get_gnrc_ft(etpy_ts, 'ref')

        return out_dict

    def _get_ecop_dens_diffs_cdfs_dict(self, probs):

        out_dict = {}
        for i, label in enumerate(self._data_ref_labels):
            for lag in self._sett_obj_lag_steps_vld:

                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs[:, i], probs[:, i], lag, True)

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
                    probs[:, i], probs[:, i], lag, True)

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

                    nth_ord_diffs_dict[(label, nth_ord)][1:] /= (
                        self._ref_nth_ord_diffs_ft_dict[(label, nth_ord)][1])

                    nth_ord_diffs_dict[(label, nth_ord)][:1] /= (
                        self._ref_nth_ord_diffs_ft_dict[(label, nth_ord)][3])

                elif vtype == 'ref':
                    pass

                else:
                    raise NotImplementedError

        return nth_ord_diffs_dict

    def _get_nth_ord_diffs_ft_dict(self, vals, nth_ords):

        nth_ords_ft_dict = self._get_nth_diff_ft_arrs(vals, nth_ords, 'ref')

        return nth_ords_ft_dict

