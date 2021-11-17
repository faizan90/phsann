'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from .base import PhaseAnnealingAlgBase as PAAB


class PhaseAnnealingAlgObjective(PAAB):

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def __init__(self, verbose=True):

        PAAB.__init__(self, verbose)
        return

    def _get_obj_scorr_val(self):

        if self._sett_obj_use_obj_dist_flag:

            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0
            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:

                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_scorr[(label, lag)])):

                        continue

                    sim_diffs = self._rs.scorr_diffs[(label, lag)]

                    ftn = self._rr.scorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val),
                            self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist
                             ) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum()

                    if self._alg_done_opt_flag:
                        self._rr.scorr_qq_dict[(label, lag)] = ref_probs
                        self._rs.scorr_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_scorr[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_scorr[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_scorr[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_scorr[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_scorr[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.scorrs - self._rs.scorrs) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_1_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0

            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:
                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_asymm_1[(label, lag)])):

                        continue

                    sim_diffs = self._rs.asymm_1_diffs[(label, lag)]

                    ftn = self._rr.asymm_1_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val),
                            self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum()

                    if self._alg_done_opt_flag:
                        self._rr.asymm_1_qq_dict[(label, lag)] = ref_probs
                        self._rs.asymm_1_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_asymm_1[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                          self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_asymm_1[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_asymm_1[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_asymm_1[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_asymm_1[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.asymms_1 - self._rs.asymms_1) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_2_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0

            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:
                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_asymm_2[(label, lag)])):

                        continue

                    sim_diffs = self._rs.asymm_2_diffs[(label, lag)].copy()

                    ftn = self._rr.asymm_2_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val
                            ), self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum()

                    if self._alg_done_opt_flag:
                        self._rr.asymm_2_qq_dict[(label, lag)] = ref_probs
                        self._rs.asymm_2_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_asymm_2[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                          self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_asymm_2[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_asymm_2[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_asymm_2[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_asymm_2[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.asymms_2 - self._rs.asymms_2) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_ecop_dens_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0
            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:

                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_ecop_dens[(label, lag)])):

                        continue

                    sim_diffs = self._rs.ecop_dens_diffs[(label, lag)]

                    ftn = self._rr.ecop_dens_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val),
                            self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum() / ftn.sclr

                    if self._alg_done_opt_flag:
                        self._rr.ecop_dens_qq_dict[(label, lag)] = ref_probs
                        self._rs.ecop_dens_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_ecop_dens[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_ecop_dens[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_ecop_dens[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_ecop_dens[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_ecop_dens[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = (
                (self._rr.ecop_dens - self._rs.ecop_dens) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_ecop_etpy_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0
            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:
                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_ecop_etpy[(label, lag)])):

                        continue

                    sim_diffs = self._rs.ecop_etpy_diffs[(label, lag)]

                    ftn = self._rr.ecop_etpy_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val
                            ), self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum() / ftn.sclr

                    if self._alg_done_opt_flag:
                        self._rr.ecop_etpy_qq_dict[(label, lag)] = ref_probs
                        self._rs.ecop_etpy_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_ecop_etpy[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_ecop_etpy[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_ecop_etpy[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_ecop_etpy[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_ecop_etpy[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.ecop_etpy - self._rs.ecop_etpy) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_nth_ord_diffs_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0
            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for nth_ord in self._sett_obj_nth_ords:

                    if (cont_flag_01_prt and
                        (not self._alg_wts_nth_order[(label, nth_ord)])):

                        continue

                    sim_diffs = self._rs.nth_ord_diffs[
                        (label, nth_ord)].copy()

                    ftn = self._rr.nth_ord_diffs_cdfs_dict[(label, nth_ord)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val),
                            self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum()

                    if self._alg_done_opt_flag:
                        self._rr.nth_ord_qq_dict[(label, nth_ord)] = (
                            ref_probs)

                        self._rs.nth_ord_qq_dict[(label, nth_ord)] = (
                            sim_probs)

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_nth_order[(label, nth_ord)]

                    elif (self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_nth_order[(label, nth_ord)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_nth_order[(label, nth_ord)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_nth_order[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_nth_order[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.nths - self._rs.nths) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_cos_sin_dist_val(self):

        obj_val = 0.0
        for i, label in enumerate(self._data_ref_labels):

            cos_ftn = self._rr.cos_sin_cdfs_dict[(label, 'cos')]
            sin_ftn = self._rr.cos_sin_cdfs_dict[(label, 'sin')]

            ref_probs_cos = cos_ftn.yr
            ref_probs_sin = sin_ftn.yr

            sim_vals_cos, sim_vals_sin = self._get_cos_sin_ift_dists(
                self._rs.ft[:, i])

            if not self._sett_obj_use_dens_ftn_flag:
                sim_probs_cos = np.maximum(np.minimum(
                    cos_ftn(sim_vals_cos),
                    self._alg_cnsts_max_prob_val),
                    self._alg_cnsts_min_prob_val)

                sim_probs_sin = np.maximum(np.minimum(
                    sin_ftn(sim_vals_sin),
                    self._alg_cnsts_max_prob_val),
                    self._alg_cnsts_min_prob_val)

                cos_sq_diffs = (
                    ((ref_probs_cos - sim_probs_cos
                      ) ** self._alg_cnsts_diffs_exp) * cos_ftn.wts)

                sin_sq_diffs = (
                    ((ref_probs_sin - sim_probs_sin
                      ) ** self._alg_cnsts_diffs_exp) * sin_ftn.wts)

            else:
                cos_sim_hist = np.histogram(
                    sim_vals_cos,
                    bins=cos_ftn.bins,
                    range=(cos_ftn.bins[0], cos_ftn.bins[-1]),
                    )[0] / sim_vals_cos.size

                cos_sq_diffs = (
                    (cos_ftn.hist - cos_sim_hist) ** self._alg_cnsts_diffs_exp)

                sin_sim_hist = np.histogram(
                    sim_vals_sin,
                    bins=sin_ftn.bins,
                    range=(sin_ftn.bins[0], sin_ftn.bins[-1]),
                    )[0] / sim_vals_sin.size

                sin_sq_diffs = (
                    (sin_ftn.hist - sin_sim_hist) ** self._alg_cnsts_diffs_exp)

            obj_val += cos_sq_diffs.sum() / cos_ftn.sclr
            obj_val += sin_sq_diffs.sum() / sin_ftn.sclr

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_pcorr_val(self):

        if self._sett_obj_use_obj_dist_flag:
            cont_flag_01_prt = (
                (not self._alg_wts_lag_nth_search_flag) and
                (self._sett_wts_lags_nths_set_flag) and
                (not self._alg_done_opt_flag))

            obj_val = 0.0
            for label in self._data_ref_labels:

                label_obj_val = 0.0
                for lag in self._sett_obj_lag_steps:
                    if (cont_flag_01_prt and
                        (not self._alg_wts_lag_pcorr[(label, lag)])):

                        continue

                    sim_diffs = self._rs.pcorr_diffs[(label, lag)]

                    ftn = self._rr.pcorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.yr

                    if ((not self._sett_obj_use_dens_ftn_flag) or
                        self._alg_done_opt_flag or (
                        self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag and
                        self._alg_cnsts_lag_wts_overall_err_flag and (
                        not self._sett_obj_use_dens_ftn_flag))):

                        sim_probs = np.maximum(np.minimum(
                            ftn(sim_diffs),
                            self._alg_cnsts_max_prob_val),
                            self._alg_cnsts_min_prob_val)

                        sim_probs_shft = self._get_penalized_probs(
                            ref_probs, sim_probs)

                        sq_diffs = (
                            (ref_probs - sim_probs_shft
                             ) ** self._alg_cnsts_diffs_exp
                            ) * ftn.wts

                    else:
                        sim_hist = np.histogram(
                            sim_diffs,
                            bins=ftn.bins,
                            range=(ftn.bins[0], ftn.bins[-1]),
                            )[0] / sim_diffs.size

                        sq_diffs = (
                            (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                    sq_diffs_sum = sq_diffs.sum()

                    if self._alg_done_opt_flag:
                        self._rr.pcorr_qq_dict[(label, lag)] = ref_probs
                        self._rs.pcorr_qq_dict[(label, lag)] = sim_probs

                    if ((not self._alg_wts_lag_nth_search_flag) and
                        (self._sett_wts_lags_nths_set_flag)):

                        wt = self._alg_wts_lag_pcorr[(label, lag)]

                    elif (self._alg_wts_lag_nth_search_flag and
                        self._sett_wts_lags_nths_set_flag):

                        if self._alg_cnsts_lag_wts_overall_err_flag and (
                            not self._sett_obj_use_dens_ftn_flag):

                            self._alg_wts_lag_pcorr[(label, lag)].append(
                                ((ref_probs - sim_probs
                                  ) ** self._alg_cnsts_diffs_exp).sum())

                        else:
                            self._alg_wts_lag_pcorr[(label, lag)].append(
                                sq_diffs_sum)

                        wt = 1

                    else:
                        wt = 1

                    label_obj_val += sq_diffs_sum * wt

                if ((not self._alg_wts_label_search_flag) and
                    (self._sett_wts_label_set_flag) and
                    (not self._alg_wts_lag_nth_search_flag)):

                    wt = self._alg_wts_label_pcorr[label]

                elif (self._alg_wts_label_search_flag and
                     (self._sett_wts_label_set_flag)  and
                     (not self._alg_wts_lag_nth_search_flag)):

                    self._alg_wts_label_pcorr[label].append(
                        label_obj_val)

                    wt = 1

                else:
                    wt = 1

                obj_val += label_obj_val * wt

        else:
            obj_val = ((self._rr.pcorrs - self._rs.pcorrs) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_1_ms_val(self):

        obj_val = 0.0
        if self._sett_obj_use_obj_dist_flag:
            for comb in self._rr.mult_asymm_1_diffs_cdfs_dict:
                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                sim_diffs = self._rs.mult_asymm_1_diffs[comb]

                ftn = self._rr.mult_asymm_1_diffs_cdfs_dict[comb]

                ref_probs = ftn.yr

                if ((not self._sett_obj_use_dens_ftn_flag) or
                    self._alg_done_opt_flag):

                    sim_probs = np.maximum(np.minimum(
                        ftn(sim_diffs),
                        self._alg_cnsts_max_prob_val),
                        self._alg_cnsts_min_prob_val)

                    sim_probs_shft = self._get_penalized_probs(
                        ref_probs, sim_probs)

                    sq_diffs = (
                        (ref_probs - sim_probs_shft
                         ) ** self._alg_cnsts_diffs_exp) * ftn.wts

                else:
                    sim_hist = np.histogram(
                        sim_diffs,
                        bins=ftn.bins,
                        range=(ftn.bins[0], ftn.bins[-1]),
                        )[0] / sim_diffs.size

                    sq_diffs = (
                        (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                obj_val += sq_diffs.sum()

                if self._alg_done_opt_flag:
                    self._rr.mult_asymm_1_qq_dict[comb] = ref_probs
                    self._rs.mult_asymm_1_qq_dict[comb] = sim_probs

        else:
            for comb in self._rr.mult_asymm_1_diffs_cdfs_dict:
                ref_diffs = (
                    self._rr.mult_asymm_1_diffs_cdfs_dict[comb].x.sum())

                sim_diffs = self._rs.mult_asymm_1_diffs[comb].sum()

                obj_val += ((ref_diffs - sim_diffs) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_2_ms_val(self):

        obj_val = 0.0
        if self._sett_obj_use_obj_dist_flag:
            for comb in self._rr.mult_asymm_2_diffs_cdfs_dict:
                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                sim_diffs = self._rs.mult_asymm_2_diffs[comb]

                ftn = self._rr.mult_asymm_2_diffs_cdfs_dict[comb]

                ref_probs = ftn.yr

                if ((not self._sett_obj_use_dens_ftn_flag) or
                    self._alg_done_opt_flag):

                    sim_probs = np.maximum(np.minimum(
                        ftn(sim_diffs),
                        self._alg_cnsts_max_prob_val),
                        self._alg_cnsts_min_prob_val)

                    sim_probs_shft = self._get_penalized_probs(
                        ref_probs, sim_probs)

                    sq_diffs = (
                        (ref_probs - sim_probs_shft
                         ) ** self._alg_cnsts_diffs_exp) * ftn.wts

                else:
                    sim_hist = np.histogram(
                        sim_diffs,
                        bins=ftn.bins,
                        range=(ftn.bins[0], ftn.bins[-1]),
                        )[0] / sim_diffs.size

                    sq_diffs = (
                        (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                obj_val += sq_diffs.sum()

                if self._alg_done_opt_flag:
                    self._rr.mult_asymm_2_qq_dict[comb] = ref_probs
                    self._rs.mult_asymm_2_qq_dict[comb] = sim_probs

        else:
            for comb in self._rr.mult_asymm_2_diffs_cdfs_dict:
                ref_diffs = (
                    self._rr.mult_asymm_2_diffs_cdfs_dict[comb].x.sum())

                sim_diffs = self._rs.mult_asymm_2_diffs[comb].sum()

                obj_val += ((ref_diffs - sim_diffs) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_ecop_dens_ms_val(self):

        obj_val = 0.0
        if self._sett_obj_use_obj_dist_flag:
            for comb in self._rr.mult_ecop_dens_cdfs_dict:
                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                sim_diffs = self._rs.mult_ecop_dens[comb]

                ftn = self._rr.mult_ecop_dens_cdfs_dict[comb]

                ref_probs = ftn.yr

                if ((not self._sett_obj_use_dens_ftn_flag) or
                    self._alg_done_opt_flag):

                    sim_probs = np.maximum(np.minimum(
                        ftn(sim_diffs),
                        self._alg_cnsts_max_prob_val),
                        self._alg_cnsts_min_prob_val)

                    sim_probs_shft = self._get_penalized_probs(
                        ref_probs, sim_probs)

                    sq_diffs = (
                        (ref_probs - sim_probs_shft
                         ) ** self._alg_cnsts_diffs_exp) * ftn.wts

                else:
                    sim_hist = np.histogram(
                        sim_diffs,
                        bins=ftn.bins,
                        range=(ftn.bins[0], ftn.bins[-1]),
                        )[0] / sim_diffs.size

                    sq_diffs = (
                        (ftn.hist - sim_hist) ** self._alg_cnsts_diffs_exp)

                obj_val += sq_diffs.sum()

                if self._alg_done_opt_flag:
                    self._rr.mult_ecop_dens_qq_dict[comb] = ref_probs
                    self._rs.mult_ecop_dens_qq_dict[comb] = sim_probs

        else:
            for comb in self._rr.mult_ecop_dens_cdfs_dict:
                ref_diffs = (
                    self._rr.mult_ecop_dens_cdfs_dict[comb].x.sum())

                sim_diffs = self._rs.mult_ecop_dens[comb].sum()

                obj_val += ((ref_diffs - sim_diffs) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_data_ft_val(self):

        obj_val = (((self._rr.data_ft - self._rs.data_ft)) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_probs_ft_val(self):

        obj_val = (((self._rr.probs_ft - self._rs.probs_ft)) ** 2).sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_1_ft_val(self):

        cont_flag_01_prt = (
            (not self._alg_wts_lag_nth_search_flag) and
            (self._sett_wts_lags_nths_set_flag) and
            (not self._alg_done_opt_flag))

        obj_val = 0.0
        for label in self._data_ref_labels:

            label_obj_val = 0.0
            for lag in self._sett_obj_lag_steps:
                if (cont_flag_01_prt and
                    (not self._alg_wts_lag_asymm_1_ft[(label, lag)])):

                    continue

                sim_ft = self._rs.asymm_1_diffs_ft[(label, lag)]

                ref_ft = self._rr.asymm_1_diffs_ft_dict[(label, lag)][0]

                sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

                sq_diffs *= self._rr.asymm_1_diffs_ft_dict[(label, lag)][2]

                sq_diffs_sum = sq_diffs.sum()

                if ((not self._alg_wts_lag_nth_search_flag) and
                    (self._sett_wts_lags_nths_set_flag)):

                    wt = self._alg_wts_lag_asymm_1_ft[(label, lag)]

                elif (self._alg_wts_lag_nth_search_flag and
                    self._sett_wts_lags_nths_set_flag):

                    self._alg_wts_lag_asymm_1_ft[(label, lag)].append(
                        sq_diffs_sum)

                    wt = 1

                else:
                    wt = 1

                label_obj_val += sq_diffs_sum * wt

            if ((not self._alg_wts_label_search_flag) and
                (self._sett_wts_label_set_flag) and
                (not self._alg_wts_lag_nth_search_flag)):

                wt = self._alg_wts_label_asymm_1_ft[label]

            elif (self._alg_wts_label_search_flag and
                 (self._sett_wts_label_set_flag)  and
                 (not self._alg_wts_lag_nth_search_flag)):

                self._alg_wts_label_asymm_1_ft[label].append(
                    label_obj_val)

                wt = 1

            else:
                wt = 1

            obj_val += label_obj_val * wt

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_2_ft_val(self):

        cont_flag_01_prt = (
            (not self._alg_wts_lag_nth_search_flag) and
            (self._sett_wts_lags_nths_set_flag) and
            (not self._alg_done_opt_flag))

        obj_val = 0.0
        for label in self._data_ref_labels:

            label_obj_val = 0.0
            for lag in self._sett_obj_lag_steps:
                if (cont_flag_01_prt and
                    (not self._alg_wts_lag_asymm_2_ft[(label, lag)])):

                    continue

                sim_ft = self._rs.asymm_2_diffs_ft[(label, lag)]

                ref_ft = self._rr.asymm_2_diffs_ft_dict[(label, lag)][0]

                sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

                sq_diffs *= self._rr.asymm_2_diffs_ft_dict[(label, lag)][2]

                sq_diffs_sum = sq_diffs.sum()

                if ((not self._alg_wts_lag_nth_search_flag) and
                    (self._sett_wts_lags_nths_set_flag)):

                    wt = self._alg_wts_lag_asymm_2_ft[(label, lag)]

                elif (self._alg_wts_lag_nth_search_flag and
                    self._sett_wts_lags_nths_set_flag):

                    self._alg_wts_lag_asymm_2_ft[(label, lag)].append(
                        sq_diffs_sum)

                    wt = 1

                else:
                    wt = 1

                label_obj_val += sq_diffs_sum * wt

            if ((not self._alg_wts_label_search_flag) and
                (self._sett_wts_label_set_flag) and
                (not self._alg_wts_lag_nth_search_flag)):

                wt = self._alg_wts_label_asymm_2_ft[label]

            elif (self._alg_wts_label_search_flag and
                 (self._sett_wts_label_set_flag)  and
                 (not self._alg_wts_lag_nth_search_flag)):

                self._alg_wts_label_asymm_2_ft[label].append(
                    label_obj_val)

                wt = 1

            else:
                wt = 1

            obj_val += label_obj_val * wt

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_nth_ord_diffs_ft_val(self):

        cont_flag_01_prt = (
            (not self._alg_wts_lag_nth_search_flag) and
            (self._sett_wts_lags_nths_set_flag) and
            (not self._alg_done_opt_flag))

        obj_val = 0.0
        for label in self._data_ref_labels:

            label_obj_val = 0.0
            for nth_ord in self._sett_obj_nth_ords:
                if (cont_flag_01_prt and
                    (not self._alg_wts_nth_order_ft[(label, nth_ord)])):

                    continue

                sim_ft = self._rs.nth_ord_diffs_ft[(label, nth_ord)]

                ref_ft = self._rr.nth_ord_diffs_ft_dict[(label, nth_ord)][0]

                sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

                sq_diffs *= self._rr.nth_ord_diffs_ft_dict[
                    (label, nth_ord)][2]

                sq_diffs_sum = sq_diffs.sum()

                if ((not self._alg_wts_lag_nth_search_flag) and
                    (self._sett_wts_lags_nths_set_flag)):

                    wt = self._alg_wts_nth_order_ft[(label, nth_ord)]

                elif (self._alg_wts_lag_nth_search_flag and
                    self._sett_wts_lags_nths_set_flag):

                    self._alg_wts_nth_order_ft[(label, nth_ord)].append(
                        sq_diffs_sum)

                    wt = 1

                else:
                    wt = 1

                label_obj_val += sq_diffs_sum * wt

            if ((not self._alg_wts_label_search_flag) and
                (self._sett_wts_label_set_flag) and
                (not self._alg_wts_lag_nth_search_flag)):

                wt = self._alg_wts_label_nth_order_ft[label]

            elif (self._alg_wts_label_search_flag and
                 (self._sett_wts_label_set_flag)  and
                 (not self._alg_wts_lag_nth_search_flag)):

                self._alg_wts_label_nth_order_ft[label].append(
                    label_obj_val)

                wt = 1

            else:
                wt = 1

            obj_val += label_obj_val * wt

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_1_ms_ft_val(self):

        sim_ft = self._rs.mult_asymm_1_cmpos_ft

        ref_ft = self._rr.mult_asymm_1_cmpos_ft_dict[0]

        sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

        sq_diffs *= self._rr.mult_asymm_1_cmpos_ft_dict[2]

        obj_val = sq_diffs.sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_asymms_2_ms_ft_val(self):

        sim_ft = self._rs.mult_asymm_2_cmpos_ft

        ref_ft = self._rr.mult_asymm_2_cmpos_ft_dict[0]

        sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

        sq_diffs *= self._rr.mult_asymm_2_cmpos_ft_dict[2]

        obj_val = sq_diffs.sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_etpy_ft_val(self):

        cont_flag_01_prt = (
            (not self._alg_wts_lag_nth_search_flag) and
            (self._sett_wts_lags_nths_set_flag) and
            (not self._alg_done_opt_flag))

        obj_val = 0.0
        for label in self._data_ref_labels:
            label_obj_wt = 0.0
            for lag in self._sett_obj_lag_steps:

                if (cont_flag_01_prt and
                    (not self._alg_wts_lag_etpy_ft[(label, lag)])):

                    continue

                sim_ft = self._rs.etpy_ft[(label, lag)]

                ref_ft = self._rr.etpy_ft_dict[(label, lag)][0]

                sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

                sq_diffs *= self._rr.etpy_ft_dict[(label, lag)][2]

                sq_diffs_sum = sq_diffs.sum()

                if ((not self._alg_wts_lag_nth_search_flag) and
                    (self._sett_wts_lags_nths_set_flag)):

                    wt = self._alg_wts_lag_etpy_ft[(label, lag)]

                elif (self._alg_wts_lag_nth_search_flag and
                    self._sett_wts_lags_nths_set_flag):

                    self._alg_wts_lag_etpy_ft[(label, lag)].append(
                        sq_diffs_sum)

                    wt = 1

                else:
                    wt = 1

                label_obj_wt += sq_diffs_sum * wt

            if ((not self._alg_wts_label_search_flag) and
                (self._sett_wts_label_set_flag) and
                (not self._alg_wts_lag_nth_search_flag)):

                wt = self._alg_wts_label_etpy_ft[label]

            elif (self._alg_wts_label_search_flag and
                 (self._sett_wts_label_set_flag)  and
                 (not self._alg_wts_lag_nth_search_flag)):

                self._alg_wts_label_etpy_ft[label].append(
                    label_obj_wt)

                wt = 1

            else:
                wt = 1

            obj_val += label_obj_wt * wt

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_obj_etpy_ms_ft_val(self):

        sim_ft = self._rs.mult_etpy_cmpos_ft

        ref_ft = self._rr.mult_etpy_cmpos_ft_dict[0]

        sq_diffs = (ref_ft - sim_ft) ** self._alg_cnsts_diffs_exp

        sq_diffs *= self._rr.mult_etpy_cmpos_ft_dict[2]

        obj_val = sq_diffs.sum()

        # So that we don't accidentally use it.
        if self._alg_done_opt_flag:
            obj_val = np.nan

        return obj_val

    def _get_penalized_probs(self, ref_probs, sim_probs):

        if self._sett_cdf_pnlt_set_flag:
            diffs = sim_probs - ref_probs

            penlt = self._sett_cdf_pnlt_n_pnlt / sim_probs.shape[0]

            lbds = sim_probs - penlt
            ubds = sim_probs + penlt

            thr = self._sett_cdf_pnlt_n_thrsh / sim_probs.shape[0]

            sim_probs_shft = sim_probs.copy()

            g_thr_idxs = (diffs > thr)
            l_ms_thr_idxs = (diffs < -thr)

            sim_probs_shft[g_thr_idxs] = ubds[g_thr_idxs]
            sim_probs_shft[l_ms_thr_idxs] = lbds[l_ms_thr_idxs]

        else:
            sim_probs_shft = sim_probs

        return sim_probs_shft

    @PAAB._timer_wrap
    def _get_obj_ftn_val(self):

        obj_vals = []

        if self._sett_obj_scorr_flag:
            obj_vals.append(self._get_obj_scorr_val())

        if self._sett_obj_asymm_type_1_flag:
            obj_vals.append(self._get_obj_asymms_1_val())

        if self._sett_obj_asymm_type_2_flag:
            obj_vals.append(self._get_obj_asymms_2_val())

        if self._sett_obj_ecop_dens_flag:
            obj_vals.append(self._get_obj_ecop_dens_val())

        if self._sett_obj_ecop_etpy_flag:
            obj_vals.append(self._get_obj_ecop_etpy_val())

        if self._sett_obj_nth_ord_diffs_flag:
            obj_vals.append(self._get_obj_nth_ord_diffs_val())

        if self._sett_obj_cos_sin_dist_flag:
            obj_vals.append(self._get_obj_cos_sin_dist_val())

        if self._sett_obj_pcorr_flag:
            obj_vals.append(self._get_obj_pcorr_val())

        if self._sett_obj_asymm_type_1_ms_flag:
            obj_vals.append(self._get_obj_asymms_1_ms_val())

        if self._sett_obj_asymm_type_2_ms_flag:
            obj_vals.append(self._get_obj_asymms_2_ms_val())

        if self._sett_obj_ecop_dens_ms_flag:
            obj_vals.append(self._get_obj_ecop_dens_ms_val())

        if self._sett_obj_match_data_ft_flag:
            obj_vals.append(self._get_obj_data_ft_val())

        if self._sett_obj_match_probs_ft_flag:
            obj_vals.append(self._get_obj_probs_ft_val())

        if self._sett_obj_asymm_type_1_ft_flag:
            obj_vals.append(self._get_obj_asymms_1_ft_val())

        if self._sett_obj_asymm_type_2_ft_flag:
            obj_vals.append(self._get_obj_asymms_2_ft_val())

        if self._sett_obj_nth_ord_diffs_ft_flag:
            obj_vals.append(self._get_obj_nth_ord_diffs_ft_val())

        if self._sett_obj_asymm_type_1_ms_ft_flag:
            obj_vals.append(self._get_obj_asymms_1_ms_ft_val())

        if self._sett_obj_asymm_type_2_ms_ft_flag:
            obj_vals.append(self._get_obj_asymms_2_ms_ft_val())

        if self._sett_obj_etpy_ft_flag:
            obj_vals.append(self._get_obj_etpy_ft_val())

        if self._sett_obj_etpy_ms_ft_flag:
            obj_vals.append(self._get_obj_etpy_ms_ft_val())

        obj_vals = np.array(obj_vals, dtype=np.float64)

        if not self._alg_done_opt_flag:
            obj_vals *= 1000

            assert np.all(np.isfinite(obj_vals)), (
                f'Invalid obj_vals: {obj_vals}!')

            if self._alg_wts_obj_search_flag:
                assert self._sett_wts_obj_wts is None

                self._alg_wts_obj_raw.append(obj_vals)

            if ((self._sett_wts_obj_wts is not None) and
                (not self._alg_done_opt_flag)):

                obj_vals *= self._sett_wts_obj_wts

        return obj_vals
