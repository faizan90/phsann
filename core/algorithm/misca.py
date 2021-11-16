'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np


class PhaseAnnealingAlgMisc:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _get_all_flags(self):

        all_flags = (
            self._sett_obj_scorr_flag,
            self._sett_obj_asymm_type_1_flag,
            self._sett_obj_asymm_type_2_flag,
            self._sett_obj_ecop_dens_flag,
            self._sett_obj_ecop_etpy_flag,
            self._sett_obj_nth_ord_diffs_flag,
            self._sett_obj_cos_sin_dist_flag,
            self._sett_obj_use_obj_dist_flag,
            self._sett_obj_use_obj_lump_flag,
            self._sett_obj_pcorr_flag,
            self._sett_obj_asymm_type_1_ms_flag,
            self._sett_obj_asymm_type_2_ms_flag,
            self._sett_obj_ecop_dens_ms_flag,
            self._sett_obj_match_data_ft_flag,
            self._sett_obj_match_probs_ft_flag,
            self._sett_obj_asymm_type_1_ft_flag,
            self._sett_obj_asymm_type_2_ft_flag,
            self._sett_obj_nth_ord_diffs_ft_flag,
            self._sett_obj_asymm_type_1_ms_ft_flag,
            self._sett_obj_asymm_type_2_ms_ft_flag,
            self._sett_obj_etpy_ft_flag,
            self._sett_obj_etpy_ms_ft_flag,
            )

        assert len(all_flags) == self._sett_obj_n_flags

        return all_flags

    def _set_all_flags_to_one_state(self, state):

        assert isinstance(state, bool), 'state not a boolean!'

        (self._sett_obj_scorr_flag,
         self._sett_obj_asymm_type_1_flag,
         self._sett_obj_asymm_type_2_flag,
         self._sett_obj_ecop_dens_flag,
         self._sett_obj_ecop_etpy_flag,
         self._sett_obj_nth_ord_diffs_flag,
         self._sett_obj_cos_sin_dist_flag,
         self._sett_obj_use_obj_dist_flag,
         self._sett_obj_use_obj_lump_flag,
         self._sett_obj_pcorr_flag,
         self._sett_obj_asymm_type_1_ms_flag,
         self._sett_obj_asymm_type_2_ms_flag,
         self._sett_obj_ecop_dens_ms_flag,
         self._sett_obj_match_data_ft_flag,
         self._sett_obj_match_probs_ft_flag,
         self._sett_obj_asymm_type_1_ft_flag,
         self._sett_obj_asymm_type_2_ft_flag,
        self._sett_obj_nth_ord_diffs_ft_flag,
        self._sett_obj_asymm_type_1_ms_ft_flag,
        self._sett_obj_asymm_type_2_ms_ft_flag,
        self._sett_obj_etpy_ft_flag,
        self._sett_obj_etpy_ms_ft_flag,
        ) = (
             [state] * self._sett_obj_n_flags)

        if self._data_ref_n_labels == 1:
            # If not the multisite case then reset to False.
            (self._sett_obj_asymm_type_1_ms_flag,
             self._sett_obj_asymm_type_2_ms_flag,
             self._sett_obj_ecop_dens_ms_flag,
             self._sett_obj_asymm_type_1_ms_ft_flag,
             self._sett_obj_asymm_type_2_ms_ft_flag,
             self._sett_obj_etpy_ms_ft_flag,) = [False] * 6

        return

    def _set_all_flags_to_mult_states(self, states):

        assert hasattr(states, '__iter__'), 'states not an iterable!'

        assert all([isinstance(state, bool) for state in states]), (
            'states has non-boolean(s)!')

        (self._sett_obj_scorr_flag,
         self._sett_obj_asymm_type_1_flag,
         self._sett_obj_asymm_type_2_flag,
         self._sett_obj_ecop_dens_flag,
         self._sett_obj_ecop_etpy_flag,
         self._sett_obj_nth_ord_diffs_flag,
         self._sett_obj_cos_sin_dist_flag,
         self._sett_obj_use_obj_dist_flag,
         self._sett_obj_use_obj_lump_flag,
         self._sett_obj_pcorr_flag,
         self._sett_obj_asymm_type_1_ms_flag,
         self._sett_obj_asymm_type_2_ms_flag,
         self._sett_obj_ecop_dens_ms_flag,
         self._sett_obj_match_data_ft_flag,
         self._sett_obj_match_probs_ft_flag,
         self._sett_obj_asymm_type_1_ft_flag,
         self._sett_obj_asymm_type_2_ft_flag,
        self._sett_obj_nth_ord_diffs_ft_flag,
        self._sett_obj_asymm_type_1_ms_ft_flag,
        self._sett_obj_asymm_type_2_ms_ft_flag,
        self._sett_obj_etpy_ft_flag,
        self._sett_obj_etpy_ms_ft_flag,
        ) = states

        if self._data_ref_n_labels == 1:
            # If not the multisite case then reset to False.
            (self._sett_obj_asymm_type_1_ms_flag,
             self._sett_obj_asymm_type_2_ms_flag,
             self._sett_obj_ecop_dens_ms_flag,
             self._sett_obj_asymm_type_1_ms_ft_flag,
             self._sett_obj_asymm_type_2_ms_ft_flag,
             self._sett_obj_etpy_ms_ft_flag,
             ) = [False] * 6

        assert len(states) == self._sett_obj_n_flags

        return

    def _update_ref_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        self._prep_vld_flag = True
        self._alg_done_opt_flag = True

        self._gen_ref_aux_data()

        self._prep_vld_flag = False
        self._alg_done_opt_flag = False

        self._set_all_flags_to_mult_states(old_flags)
        return

    def _update_sim_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        self._prep_vld_flag = True

        self._rs.ft = self._rs.ft_best.copy()

        self._rs.phs_spec = np.angle(self._rs.ft)
        self._rs.mag_spec = np.abs(self._rs.ft)

        self._rr.scorr_qq_dict = {}
        self._rr.asymm_1_qq_dict = {}
        self._rr.asymm_2_qq_dict = {}
        self._rr.ecop_dens_qq_dict = {}
        self._rr.ecop_etpy_qq_dict = {}
        self._rr.nth_ord_qq_dict = {}
        self._rr.pcorr_qq_dict = {}

        self._rr.mult_asymm_1_qq_dict = {}
        self._rr.mult_asymm_2_qq_dict = {}
        self._rr.mult_ecop_dens_qq_dict = {}

        self._rs.scorr_qq_dict = {}
        self._rs.asymm_1_qq_dict = {}
        self._rs.asymm_2_qq_dict = {}
        self._rs.ecop_dens_qq_dict = {}
        self._rs.ecop_etpy_qq_dict = {}
        self._rs.nth_ord_qq_dict = {}
        self._rs.pcorr_qq_dict = {}

        self._rs.mult_asymm_1_qq_dict = {}
        self._rs.mult_asymm_2_qq_dict = {}
        self._rs.mult_ecop_dens_qq_dict = {}

        old_lags = self._sett_obj_lag_steps.copy()
        old_nths = self._sett_obj_nth_ords.copy()

        self._sett_obj_lag_steps = self._sett_obj_lag_steps_vld
        self._sett_obj_nth_ords = self._sett_obj_nth_ords_vld

        old_lag_nth_wts_flag = self._sett_wts_lags_nths_set_flag
        old_label_wts_flag = self._sett_wts_label_set_flag

        self._sett_wts_lags_nths_set_flag = False
        self._sett_wts_label_set_flag = False
        self._alg_done_opt_flag = True

        self._update_sim_no_prms()

        self._get_obj_ftn_val()

        self._alg_done_opt_flag = False
        self._sett_wts_lags_nths_set_flag = old_lag_nth_wts_flag
        self._sett_wts_label_set_flag = old_label_wts_flag

        self._sett_obj_lag_steps = old_lags
        self._sett_obj_nth_ords = old_nths

        self._prep_vld_flag = False

        self._set_all_flags_to_mult_states(old_flags)
        return
