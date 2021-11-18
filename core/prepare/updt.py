'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from itertools import product

import numpy as np

from fcopulas import (
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    asymms_exp,
    fill_cumm_dist_from_bivar_emp_dens,
    get_srho_plus_for_probs_nd,
    get_srho_minus_for_probs_nd,
    get_etpy_nd_from_probs,
    )

from .cdfs import PhaseAnnealingPrepareCDFS as PAPCDFS
from ...misc import (
    roll_real_2arrs,
    get_local_entropy_ts_cy,
    )


class PhaseAnnealingPrepareUpdate(PAPCDFS):

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self, verbose=True):

        PAPCDFS.__init__(self, verbose)
        return

    @PAPCDFS._timer_wrap
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
            probs = self._rr.probs
            data = self._rr.data

            rltzn_cls = self._rr

        elif vtype == 'sim':
            probs = self._rs.probs
            data = self._rs.data

            rltzn_cls = self._rs

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

            # First row and first col are temps. They are always zeros.
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
            (self._rr.mult_asymm_1_diffs_cdfs_dict is not None)):

            mult_asymm_1_diffs = {}

        else:
            mult_asymm_1_diffs = None

        if ((vtype == 'sim') and
            (self._rr.mult_asymm_2_diffs_cdfs_dict is not None)):

            mult_asymm_2_diffs = {}

        else:
            mult_asymm_2_diffs = None

        if ((vtype == 'sim') and
            (self._rr.mult_ecop_dens_cdfs_dict is not None)):

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
                data_ft_norm_vals = self._rr.data_ft_norm_vals

            else:
                data_ft_norm_vals = None

            data_ft = self._get_data_ft(data, vtype, data_ft_norm_vals)

        else:
            data_ft = None

        if self._sett_obj_match_probs_ft_flag:
            if vtype == 'sim':
                probs_ft_norm_vals = self._rr.probs_ft_norm_vals

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

        if self._sett_obj_etpy_ft_flag:
            if vtype == 'sim':
                etpy_ft = {}

                if ((cont_flag_01_prt) and
                    (self._alg_wts_lag_etpy_ft is not None)):

                    etpy_ft_conts = {
                        (label, lag):
                        bool(self._alg_wts_lag_etpy_ft[(label, lag)])
                        for lag in lag_steps
                        for label in self._data_ref_labels}

                else:
                    etpy_ft_conts = {}

            else:
                etpy_ft = None

        else:
            etpy_ft = None

        if self._sett_obj_scorr_ms_flag:
            scorrs_ms = np.array(
                [get_srho_plus_for_probs_nd(probs.copy(order='c')),
                 get_srho_minus_for_probs_nd(probs.copy(order='c'))],
                dtype=np.float64)

        else:
            scorrs_ms = None

        if self._sett_obj_etpy_ms_flag:
            ecop_etpy_ms = np.array(
                [get_etpy_nd_from_probs(
                    probs.copy(order='c'), self._sett_obj_ecop_dens_bins)],
                dtype=np.float64)

            ecop_etpy_ms -= self._get_etpy_min_nd(
                self._sett_obj_ecop_dens_bins)

            ecop_etpy_ms /= (
                self._get_etpy_max_nd(self._sett_obj_ecop_dens_bins) -
                self._get_etpy_min_nd(self._sett_obj_ecop_dens_bins))

        else:
            ecop_etpy_ms = None

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
        c_etpy_ft = etpy_ft is not None

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
            c_asymm_2_diffs_ft,
            c_etpy_ft])

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
                    probs[:, j], probs[:, j], lag, True)

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

                    asymm_1_diffs_ft[(label, lag)][1:] /= (
                        self._rr.asymm_1_diffs_ft_dict[(label, lag)])[1]

                    asymm_1_diffs_ft[(label, lag)][:1] /= (
                        self._rr.asymm_1_diffs_ft_dict[(label, lag)])[3]

                if (c_asymm_2_diffs_ft and
                    asymm_2_diff_ft_conts.get((label, lag), True)):

                    asymm_2_diffs_ft[(label, lag)] = self._get_gnrc_ft(
                        (probs_i - rolled_probs_i) ** asymms_exp, 'sim')[0]

                    asymm_2_diffs_ft[(label, lag)][1:] /= (
                        self._rr.asymm_2_diffs_ft_dict[(label, lag)])[1]

                    asymm_2_diffs_ft[(label, lag)][:1] /= (
                        self._rr.asymm_2_diffs_ft_dict[(label, lag)])[3]

                if (c_etpy_ft and etpy_ft_conts.get((label, lag), True)):

                    etpy_ts = get_local_entropy_ts_cy(
                        probs_i,
                        rolled_probs_i,
                        self._sett_obj_ecop_dens_bins)

                    assert np.all(np.isfinite(etpy_ts))

                    etpy_ft[(label, lag)] = self._get_gnrc_ft(
                        etpy_ts, 'sim')[0]

                    etpy_ft[(label, lag)][1:] /= self._rr.etpy_ft_dict[
                        (label, lag)][1]

                    etpy_ft[(label, lag)][:1] /= self._rr.etpy_ft_dict[
                        (label, lag)][3]

            if cb:
                if ((c_pcorrs) or
                    (c_pcorr_diffs and
                     pcorr_diff_conts.get((label, lag), True))):

                    data_i, rolled_data_i = roll_real_2arrs(
                        data[:, j], data[:, j], lag, False)

                if c_pcorrs:
                    pcorrs[j, i] = np.corrcoef(data_i, rolled_data_i)[0, 1]

                if c_pcorr_diffs and pcorr_diff_conts.get((label, lag), True):
                    pcorr_diffs[(label, lag)] = np.sort(
                        (data_i - rolled_data_i))

        if mult_asymm_1_diffs is not None:
            for comb in self._rr.mult_asymm_1_diffs_cdfs_dict:
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
            for comb in self._rr.mult_asymm_2_diffs_cdfs_dict:
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
            for comb in self._rr.mult_ecop_dens_cdfs_dict:
                col_idxs = [
                    self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError(
                        'Ecop density configured for pairs only!')

                fill_bi_var_cop_dens(
                    probs[:, col_idxs[0]],
                    probs[:, col_idxs[1]],
                    mult_ecop_dens_arr)

                mult_ecop_dens_diffs[comb] = np.sort(
                    mult_ecop_dens_arr.ravel())

        if self._sett_obj_asymm_type_1_ms_ft_flag and (vtype == 'sim'):
            mult_asymm_1_cmpos_ft = self._get_mult_asymm_1_cmpos_ft(
                probs, vtype)[0]

            mult_asymm_1_cmpos_ft[
                self._rr.mult_asymm_1_cmpos_ft_dict[3]:] /= (
                    self._rr.mult_asymm_1_cmpos_ft_dict[1])

            mult_asymm_1_cmpos_ft[:
                self._rr.mult_asymm_1_cmpos_ft_dict[3]] /= (
                    self._rr.mult_asymm_1_cmpos_ft_dict[4])

        elif vtype == 'ref':
            pass

        else:
            mult_asymm_1_cmpos_ft = None

        if self._sett_obj_asymm_type_2_ms_ft_flag and (vtype == 'sim'):
            mult_asymm_2_cmpos_ft = self._get_mult_asymm_2_cmpos_ft(
                probs, vtype)[0]

            mult_asymm_2_cmpos_ft[
                self._rr.mult_asymm_2_cmpos_ft_dict[3]:] /= (
                    self._rr.mult_asymm_2_cmpos_ft_dict[1])

            mult_asymm_2_cmpos_ft[:
                self._rr.mult_asymm_2_cmpos_ft_dict[3]] /= (
                    self._rr.mult_asymm_2_cmpos_ft_dict[4])

        elif vtype == 'ref':
            pass

        else:
            mult_asymm_2_cmpos_ft = None

        if self._sett_obj_etpy_ms_ft_flag and (vtype == 'sim'):
            mult_etpy_cmpos_ft = self._get_mult_etpy_cmpos_ft(
                probs, vtype)[0]

            mult_etpy_cmpos_ft[
                self._rr.mult_etpy_cmpos_ft_dict[3]:] /= (
                    self._rr.mult_etpy_cmpos_ft_dict[1])

            mult_etpy_cmpos_ft[:
                self._rr.mult_etpy_cmpos_ft_dict[3]] /= (
                    self._rr.mult_etpy_cmpos_ft_dict[4])

        elif vtype == 'ref':
            pass

        else:
            mult_etpy_cmpos_ft = None

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

        # Why did I do this?
        # Probably because this is only needed by asymms otherwise?
        if not self._sett_obj_scorr_flag:
            scorrs = None

        # NOTE: Update the snapshot method in Algorithm accordingly.
        rltzn_cls.scorrs = scorrs
        rltzn_cls.asymms_1 = asymms_1
        rltzn_cls.asymms_2 = asymms_2
        rltzn_cls.ecop_dens = ecop_cumm_dens_arrs  # ecop_dens_arrs
        rltzn_cls.ecop_etpy = ecop_etpy_arrs
        rltzn_cls.pcorrs = pcorrs
        rltzn_cls.nths = nths
        rltzn_cls.data_ft = data_ft
        rltzn_cls.probs_ft = probs_ft
        rltzn_cls.scorrs_ms = scorrs_ms
        rltzn_cls.ecop_etpy_ms = ecop_etpy_ms

        if vtype == 'ref':
            pass

        elif vtype == 'sim':
            self._rs.scorr_diffs = scorr_diffs
            self._rs.asymm_1_diffs = asymm_1_diffs
            self._rs.asymm_2_diffs = asymm_2_diffs
            self._rs.ecop_dens_diffs = ecop_dens_diffs
            self._rs.ecop_etpy_diffs = ecop_etpy_diffs
            self._rs.nth_ord_diffs = nth_ord_diffs
            self._rs.pcorr_diffs = pcorr_diffs

            self._rs.asymm_1_diffs_ft = asymm_1_diffs_ft
            self._rs.asymm_2_diffs_ft = asymm_2_diffs_ft
            self._rs.nth_ord_diffs_ft = nth_ord_diffs_ft
            self._rs.etpy_ft = etpy_ft

            self._rs.mult_asymm_1_diffs = mult_asymm_1_diffs
            self._rs.mult_asymm_2_diffs = mult_asymm_2_diffs
            self._rs.mult_ecop_dens = mult_ecop_dens_diffs

            self._rs.mult_asymm_1_cmpos_ft = mult_asymm_1_cmpos_ft
            self._rs.mult_asymm_2_cmpos_ft = mult_asymm_2_cmpos_ft
            self._rs.mult_etpy_cmpos_ft = mult_etpy_cmpos_ft

        else:
            raise ValueError(f'Unknown vtype in _update_obj_vars: {vtype}!')

        return
