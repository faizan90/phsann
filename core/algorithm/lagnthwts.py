'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np
from scipy.stats import norm

from ...misc import sci_round
from ..prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgLagNthWts:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    @PAP._timer_wrap
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

    def _get_scaled_wts(self, mean_obj_vals):

        movs_sum = mean_obj_vals.sum()

        wts = mean_obj_vals / movs_sum

        wts **= self._sett_wts_lags_nths_exp

        wts_sclr = movs_sum / (mean_obj_vals * wts).sum()

        wts *= wts_sclr

        return wts

    def _update_lag_nth_wt(self, labels, lags_nths, lag_nth_dict):

        '''
        Based on:
        (wts * mean_obj_vals).sum() == mean_obj_vals.sum()

        Smaller obj vals get smaller weights.
        '''

        wts_nn = self._sett_wts_lags_nths_cumm_wts_contrib is not None
        n_thresh_nn = self._sett_wts_lags_nths_n_thresh is not None

        for label in labels:

            if len(lags_nths) > 1:

                mean_obj_vals = []
                for lag_nth in lags_nths:

                    assert len(lag_nth_dict[(label, lag_nth)])

                    mean_obj_val = np.array(
                        lag_nth_dict[(label, lag_nth)]).min()

                    mean_obj_vals.append(mean_obj_val)

                mean_obj_vals = np.array(mean_obj_vals)

                assert np.all(np.isfinite(mean_obj_vals))

                wts = self._get_scaled_wts(mean_obj_vals)

                if wts_nn or n_thresh_nn:
                    norm = wts / wts.sum()
                    sort_idxs = np.argsort(norm)[::-1]
                    sort = norm[sort_idxs]

                    if (wts_nn and
                        (self._sett_wts_lags_nths_cumm_wts_contrib < 1)):

                        cumsum = sort.cumsum()

                        excld_idxs = ((
                            cumsum -
                            self._sett_wts_lags_nths_cumm_wts_contrib) >= 0)

                        cut_off_idx_wts_lim = np.where(excld_idxs)[0][0]

                    else:
                        cut_off_idx_wts_lim = mean_obj_vals.size

                    if n_thresh_nn:
                        cut_off_idx_n_thresh = (
                            self._sett_wts_lags_nths_n_thresh - 1)

                    else:
                        cut_off_idx_n_thresh = mean_obj_vals.size

                    cut_off_idx = min(
                        cut_off_idx_n_thresh, cut_off_idx_wts_lim)

                    assert cut_off_idx >= 0

                    mean_obj_vals[sort_idxs[cut_off_idx + 1:]] = 0

                    wts = self._get_scaled_wts(mean_obj_vals)

                assert np.all(wts >= 0)
                assert wts.sum()

                assert np.isclose(
                    (wts * mean_obj_vals).sum(), mean_obj_vals.sum())

                wts = sci_round(wts)

            else:
                wts = [1.0]

            for i, lag in enumerate(lags_nths):
                lag_nth_dict[(label, lag)] = wts[i]

        return

    def _update_lag_nth_wts(self):

        if self._sett_obj_scorr_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_scorr)

        if self._sett_obj_asymm_type_1_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_1)

        if self._sett_obj_asymm_type_2_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_2)

        if self._sett_obj_ecop_dens_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_ecop_dens)

        if self._sett_obj_ecop_etpy_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_ecop_etpy)

        if self._sett_obj_nth_ord_diffs_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_nth_ords,
                self._alg_wts_nth_order)

        if self._sett_obj_pcorr_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_pcorr)

        if self._sett_obj_asymm_type_1_ft_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_1_ft)

        if self._sett_obj_asymm_type_2_ft_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_2_ft)

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_nth_ords,
                self._alg_wts_nth_order_ft)

        if self._sett_obj_etpy_ft_flag:
            self._update_lag_nth_wt(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_etpy_ft)

        return

    def _fill_lag_nth_dict(self, labels, lags_nths, lag_nth_dict):

        for label in labels:
            for lag_nth in lags_nths:
                lag_nth_dict[(label, lag_nth)] = []

        return

    def _init_lag_nth_wts(self):

        any_obj_ftn = False

        if self._sett_obj_scorr_flag:
            self._alg_wts_lag_scorr = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_scorr)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_1_flag:
            self._alg_wts_lag_asymm_1 = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_1)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_2_flag:
            self._alg_wts_lag_asymm_2 = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_2)

            any_obj_ftn = True

        if self._sett_obj_ecop_dens_flag:
            self._alg_wts_lag_ecop_dens = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_ecop_dens)

            any_obj_ftn = True

        if self._sett_obj_ecop_etpy_flag:
            self._alg_wts_lag_ecop_etpy = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_ecop_etpy)

            any_obj_ftn = True

        if self._sett_obj_nth_ord_diffs_flag:
            self._alg_wts_nth_order = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_nth_ords,
                self._alg_wts_nth_order)

            any_obj_ftn = True

        if self._sett_obj_pcorr_flag:
            self._alg_wts_lag_pcorr = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_pcorr)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_1_ft_flag:
            self._alg_wts_lag_asymm_1_ft = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_1_ft)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_2_ft_flag:
            self._alg_wts_lag_asymm_2_ft = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_asymm_2_ft)

            any_obj_ftn = True

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._alg_wts_nth_order_ft = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_nth_ords,
                self._alg_wts_nth_order_ft)

            any_obj_ftn = True

        if self._sett_obj_etpy_ft_flag:
            self._alg_wts_lag_etpy_ft = {}

            self._fill_lag_nth_dict(
                self._data_ref_labels,
                self._sett_obj_lag_steps,
                self._alg_wts_lag_etpy_ft)

            any_obj_ftn = True

        assert any_obj_ftn, (
            'None of the objective functions involved lags and nth_ords '
            'weights are active!')

        return
