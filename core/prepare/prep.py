'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from collections import namedtuple
from itertools import product

import numpy as np
from scipy.stats import norm

from fcopulas import (
    get_asymm_1_sample,
    get_asymm_2_sample,
    fill_bi_var_cop_dens,
    asymms_exp,
    fill_cumm_dist_from_bivar_emp_dens,
    )

from .tfms import PhaseAnnealingPrepareTfms
from .cdfs import PhaseAnnealingPrepareCDFS
from ..settings import PhaseAnnealingSettings as PAS
from ...misc import (
    print_sl,
    print_el,
    roll_real_2arrs,
    get_local_entropy_ts_cy,
    )


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

        self._data_tfm_type = 'probs'
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
        self._ref_mult_ecop_dens_cdfs_dict = None

        self._ref_scorr_qq_dict = None
        self._ref_asymm_1_qq_dict = None
        self._ref_asymm_2_qq_dict = None
        self._ref_ecop_dens_qq_dict = None
        self._ref_ecop_etpy_qq_dict = None
        self._ref_nth_ord_qq_dict = None
        self._ref_pcorr_qq_dict = None

        self._ref_mult_asymm_1_qq_dict = None
        self._ref_mult_asymm_2_qq_dict = None
        self._ref_mult_etpy_dens_qq_dict = None

        self._ref_asymm_1_diffs_ft_dict = None
        self._ref_asymm_2_diffs_ft_dict = None
        self._ref_nth_ord_diffs_ft_dict = None
        self._ref_etpy_ft_dict = None
        self._ref_mult_asymm_1_cmpos_ft_dict = None
        self._ref_mult_asymm_2_cmpos_ft_dict = None
        self._ref_mult_etpy_cmpos_ft_dict = None

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
        self._sim_mult_ecop_dens = None

        self._sim_asymm_1_diffs_ft = None
        self._sim_asymm_2_diffs_ft = None
        self._sim_nth_ord_diffs_ft = None
        self._sim_etpy_ft = None
        self._sim_mult_asymm_1_cmpos_ft = None
        self._sim_mult_asymm_2_cmpos_ft = None
        self._sim_mult_etpy_cmpos_ft = None

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
        self._sim_mult_ecop_dens_qq_dict = None  # TODO

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
            (self._ref_mult_ecop_dens_cdfs_dict is not None)):

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
                        self._ref_asymm_1_diffs_ft_dict[(label, lag)])[1]

                    asymm_1_diffs_ft[(label, lag)][:1] /= (
                        self._ref_asymm_1_diffs_ft_dict[(label, lag)])[3]

                if (c_asymm_2_diffs_ft and
                    asymm_2_diff_ft_conts.get((label, lag), True)):

                    asymm_2_diffs_ft[(label, lag)] = self._get_gnrc_ft(
                        (probs_i - rolled_probs_i) ** asymms_exp, 'sim')[0]

                    asymm_2_diffs_ft[(label, lag)][1:] /= (
                        self._ref_asymm_2_diffs_ft_dict[(label, lag)])[1]

                    asymm_2_diffs_ft[(label, lag)][:1] /= (
                        self._ref_asymm_2_diffs_ft_dict[(label, lag)])[3]

                if (c_etpy_ft and etpy_ft_conts.get((label, lag), True)):

                    etpy_ts = get_local_entropy_ts_cy(
                        probs_i,
                        rolled_probs_i,
                        self._sett_obj_ecop_dens_bins)

                    assert np.all(np.isfinite(etpy_ts))

                    etpy_ft[(label, lag)] = self._get_gnrc_ft(
                        etpy_ts, 'sim')[0]

                    etpy_ft[(label, lag)][1:] /= self._ref_etpy_ft_dict[
                        (label, lag)][1]

                    etpy_ft[(label, lag)][:1] /= self._ref_etpy_ft_dict[
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
            for comb in self._ref_mult_ecop_dens_cdfs_dict:
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
                self._ref_mult_asymm_1_cmpos_ft_dict[3]:] /= (
                    self._ref_mult_asymm_1_cmpos_ft_dict[1])

            mult_asymm_1_cmpos_ft[:
                self._ref_mult_asymm_1_cmpos_ft_dict[3]] /= (
                    self._ref_mult_asymm_1_cmpos_ft_dict[4])

        elif vtype == 'ref':
            pass

        else:
            mult_asymm_1_cmpos_ft = None

        if self._sett_obj_asymm_type_2_ms_ft_flag and (vtype == 'sim'):
            mult_asymm_2_cmpos_ft = self._get_mult_asymm_2_cmpos_ft(
                probs, vtype)[0]

            mult_asymm_2_cmpos_ft[
                self._ref_mult_asymm_2_cmpos_ft_dict[3]:] /= (
                    self._ref_mult_asymm_2_cmpos_ft_dict[1])

            mult_asymm_2_cmpos_ft[:
                self._ref_mult_asymm_2_cmpos_ft_dict[3]] /= (
                    self._ref_mult_asymm_2_cmpos_ft_dict[4])

        elif vtype == 'ref':
            pass

        else:
            mult_asymm_2_cmpos_ft = None

        if self._sett_obj_etpy_ms_ft_flag and (vtype == 'sim'):
            mult_etpy_cmpos_ft = self._get_mult_etpy_cmpos_ft(
                probs, vtype)[0]

            mult_etpy_cmpos_ft[
                self._ref_mult_etpy_cmpos_ft_dict[3]:] /= (
                    self._ref_mult_etpy_cmpos_ft_dict[1])

            mult_etpy_cmpos_ft[:
                self._ref_mult_etpy_cmpos_ft_dict[3]] /= (
                    self._ref_mult_etpy_cmpos_ft_dict[4])

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
            self._sim_etpy_ft = etpy_ft

            self._sim_mult_asymms_1_diffs = mult_asymm_1_diffs
            self._sim_mult_asymms_2_diffs = mult_asymm_2_diffs
            self._sim_mult_ecop_dens = mult_ecop_dens_diffs

            self._sim_mult_asymm_1_cmpos_ft = mult_asymm_1_cmpos_ft
            self._sim_mult_asymm_2_cmpos_ft = mult_asymm_2_cmpos_ft
            self._sim_mult_etpy_cmpos_ft = mult_etpy_cmpos_ft

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

#         import matplotlib.pyplot as plt
#
#         for i in range(self._ref_ft.shape[1]):
#             plt.figure(figsize=(7, 7))
#
#             plt.scatter(
#                 rankdata(self._ref_ft[:, i].real) / (self._ref_ft.shape[0] + 1),
#                 rankdata(self._ref_ft[:, i].imag) / (self._ref_ft.shape[0] + 1),
#                 alpha=0.5)
#
#             plt.grid()
#             plt.gca().set_axisbelow(True)
#
#             plt.gca().set_aspect('equal')
#
#             plt.savefig(
#                 f'P:/Downloads/test_ft_ref_ecop_{i}.png')
#
#             plt.close()

#         import matplotlib.pyplot as plt
#
#         for i in range(self._ref_ft.shape[1]):
#             plt.figure(figsize=(7, 7))
#
#             plt.scatter(
#                 rankdata(np.cos(self._ref_phs_spec[+1:, i])) / (self._ref_ft.shape[0] + 1),
#                 rankdata(np.cos(self._ref_phs_spec[:-1, i])) / (self._ref_ft.shape[0] + 1),
#                 alpha=0.5)
#
#             plt.grid()
#             plt.gca().set_axisbelow(True)
#
#             plt.gca().set_aspect('equal')
#
#             plt.savefig(
#                 f'P:/Downloads/test_phs_ref_ecop_{i}.png')
#
#             plt.close()

#         raise Exception

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

        if self._sett_obj_etpy_ft_flag:
            self._ref_etpy_ft_dict = (
                self._get_etpy_ft_dict(self._ref_probs))

        if self._data_ref_n_labels > 1:
            # NOTE: don't add flags here
            self._ref_mult_asymm_1_diffs_cdfs_dict = (
                self._get_mult_asymm_1_diffs_cdfs_dict(self._ref_probs))

            self._ref_mult_asymm_2_diffs_cdfs_dict = (
                self._get_mult_asymm_2_diffs_cdfs_dict(self._ref_probs))

            self._ref_mult_ecop_dens_cdfs_dict = (
                self._get_mult_ecop_dens_diffs_cdfs_dict(self._ref_probs))

            self._ref_mult_asymm_1_cmpos_ft_dict = (
                self._get_mult_asymm_1_cmpos_ft(self._ref_probs, 'ref'))

            self._ref_mult_asymm_2_cmpos_ft_dict = (
                self._get_mult_asymm_2_cmpos_ft(self._ref_probs, 'ref'))

            self._ref_mult_etpy_cmpos_ft_dict = (
                self._get_mult_etpy_cmpos_ft(self._ref_probs, 'ref'))

#             self._get_mult_scorr_cmpos_ft(self._ref_probs, 'ref')

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

#         self._get_mult_asymm_1_cmpos_ft(self._sim_probs, 'sim')
#         self._get_mult_asymm_2_cmpos_ft(self._sim_probs, 'sim')
#         self._get_mult_scorr_cmpos_ft(self._sim_probs, 'sim')

#         import matplotlib.pyplot as plt
#
#         for i in range(self._sim_ft.shape[1]):
#             plt.figure(figsize=(7, 7))
#
#             plt.scatter(
#                 rankdata(self._sim_ft[:, i].real) / (self._sim_ft.shape[0] + 1),
#                 rankdata(self._sim_ft[:, i].imag) / (self._sim_ft.shape[0] + 1),
#                 alpha=0.5)
#
#             plt.grid()
#             plt.gca().set_axisbelow(True)
#
#             plt.gca().set_aspect('equal')
#
#             plt.savefig(
#                 f'P:/Downloads/test_ft_sim_ecop_{i}.png')
#
#             plt.close()
#

#         import matplotlib.pyplot as plt
#
#         for i in range(self._sim_ft.shape[1]):
#             plt.figure(figsize=(7, 7))
#
#             plt.scatter(
#                 rankdata(np.cos(self._sim_phs_spec[+1:, i])) / (self._sim_ft.shape[0] + 1),
#                 rankdata(np.cos(self._sim_phs_spec[:-1, i])) / (self._sim_ft.shape[0] + 1),
#                 alpha=0.5)
#
#             plt.grid()
#             plt.gca().set_axisbelow(True)
#
#             plt.gca().set_aspect('equal')
#
#             plt.savefig(
#                 f'P:/Downloads/test_phs_sim_ecop_{i}.png')
#
#             plt.close()

#         raise Exception

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

        sim_rltzns_out_labs.extend(
            [f'etpy_ft_{label}_{lag:03d}'
             for label in self._data_ref_labels
             for lag in self._sett_obj_lag_steps_vld])

        if self._data_ref_n_labels > 1:
            sim_rltzns_out_labs.extend(
                [f'mult_asymm_1_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_1_diffs_cdfs_dict])

            sim_rltzns_out_labs.extend(
                [f'mult_asymm_2_diffs_{"_".join(comb)}'
                 for comb in self._ref_mult_asymm_2_diffs_cdfs_dict])

            sim_rltzns_out_labs.extend(
                [f'mult_ecop_dens_{"_".join(comb)}'
                 for comb in self._ref_mult_ecop_dens_cdfs_dict])

            sim_rltzns_out_labs.append('mult_asymm_1_cmpos_ft')
            sim_rltzns_out_labs.append('mult_asymm_2_cmpos_ft')
            sim_rltzns_out_labs.append('mult_etpy_cmpos_ft')

        # Same for QQ probs.
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

            sim_rltzns_out_labs.extend(
                [f'mult_ecop_dens_qq_{"_".join(comb)}'
                 for comb in self._ref_mult_ecop_dens_cdfs_dict])

        # Initialize.
        self._sim_rltzns_proto_tup = namedtuple(
            'SimRltznData', sim_rltzns_out_labs)

        if self._vb:
            print_sl()

            print(f'Phase annealing preparation done successfully!')

            print_el()

        self._prep_verify_flag = True
        return

    __verify = verify
