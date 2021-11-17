'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from ...misc import sci_round
from .lagnthwts import PhaseAnnealingAlgLagNthWts as PAALNW


class PhaseAnnealingAlgLabelWts(PAALNW):

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self, verbose=True):

        PAALNW.__init__(self, verbose)
        return

    @PAALNW._timer_wrap
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

    def _update_label_wt(self, labels, label_dict):

        '''
        Based on:
        (wts * mean_obj_vals).sum() == mean_obj_vals.sum()
        '''

        # min seems to work better.
        mean_obj_vals = []
        for label in labels:
            assert len(label_dict[label])

            mean_obj_val = np.array(label_dict[label]).min()
            mean_obj_vals.append(mean_obj_val)

        mean_obj_vals = np.array(mean_obj_vals)

        assert np.all(np.isfinite(mean_obj_vals))

        movs_sum = mean_obj_vals.sum()

        # The line that is different than lags_nths.
        wts = mean_obj_vals / movs_sum

        wts **= self._sett_wts_label_exp

        wts_sclr = movs_sum / (mean_obj_vals * wts).sum()

        wts *= wts_sclr

        assert np.isclose((wts * mean_obj_vals).sum(), mean_obj_vals.sum())

        wts = sci_round(wts)

        for i, label in enumerate(labels):
            label_dict[label] = wts[i]

        return

    def _update_label_wts(self):

        if self._sett_obj_scorr_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_scorr)

        if self._sett_obj_asymm_type_1_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_asymm_1)

        if self._sett_obj_asymm_type_2_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_asymm_2)

        if self._sett_obj_ecop_dens_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_ecop_dens)

        if self._sett_obj_ecop_etpy_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_ecop_etpy)

        if self._sett_obj_nth_ord_diffs_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_nth_order)

        if self._sett_obj_pcorr_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_pcorr)

        if self._sett_obj_asymm_type_1_ft_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_asymm_1_ft)

        if self._sett_obj_asymm_type_2_ft_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_asymm_2_ft)

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_nth_order_ft)

        if self._sett_obj_etpy_ft_flag:
            self._update_label_wt(
                self._data_ref_labels,
                self._alg_wts_label_etpy_ft)

        return

    def _fill_label_dict(self, labels, label_dict):

        for label in labels:
            label_dict[label] = []

        return

    def _init_label_wts(self):

        any_obj_ftn = False

        if self._sett_obj_scorr_flag:
            self._alg_wts_label_scorr = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_scorr)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_1_flag:
            self._alg_wts_label_asymm_1 = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_asymm_1)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_2_flag:
            self._alg_wts_label_asymm_2 = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_asymm_2)

            any_obj_ftn = True

        if self._sett_obj_ecop_dens_flag:
            self._alg_wts_label_ecop_dens = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_ecop_dens)

            any_obj_ftn = True

        if self._sett_obj_ecop_etpy_flag:
            self._alg_wts_label_ecop_etpy = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_ecop_etpy)

            any_obj_ftn = True

        if self._sett_obj_nth_ord_diffs_flag:
            self._alg_wts_label_nth_order = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_nth_order)

            any_obj_ftn = True

        if self._sett_obj_pcorr_flag:
            self._alg_wts_label_pcorr = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_pcorr)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_1_ft_flag:
            self._alg_wts_label_asymm_1_ft = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_asymm_1_ft)

            any_obj_ftn = True

        if self._sett_obj_asymm_type_2_ft_flag:
            self._alg_wts_label_asymm_2_ft = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_asymm_2_ft)

            any_obj_ftn = True

        if self._sett_obj_nth_ord_diffs_ft_flag:
            self._alg_wts_label_nth_order_ft = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_nth_order_ft)

            any_obj_ftn = True

        if self._sett_obj_etpy_ft_flag:
            self._alg_wts_label_etpy_ft = {}

            self._fill_label_dict(
                self._data_ref_labels,
                self._alg_wts_label_etpy_ft)

            any_obj_ftn = True

        assert any_obj_ftn, (
            'None of the objective functions involved in label weights are '
            'active!')

        return
