'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from collections import namedtuple

import numpy as np
from scipy.stats import norm

from .tfms import PhaseAnnealingPrepareTfms
from .cdfs import PhaseAnnealingPrepareCDFS
from .updt import PhaseAnnealingPrepareUpdate
from ..settings import PhaseAnnealingSettings as PAS
from ...misc import (
    print_sl,
    print_el,
    )


class PhaseAnnealingPrepare(
        PAS,
        PhaseAnnealingPrepareTfms,
        PhaseAnnealingPrepareCDFS,
        PhaseAnnealingPrepareUpdate):

    '''Prepare derived variables required by phase annealing here.'''

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
