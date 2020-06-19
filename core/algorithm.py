'''
Created on Dec 27, 2019

@author: Faizan
'''
from time import asctime
from fnmatch import fnmatch
from collections import deque
from timeit import default_timer

import h5py
import numpy as np
from scipy.interpolate import interp1d
from multiprocessing import Manager, Lock
from pathos.multiprocessing import ProcessPool

from ..misc import print_sl, print_el, ret_mp_idxs
from .prepare import PhaseAnnealingPrepare as PAP

trunc_interp_ftns_flag = False

diffs_exp = 1.0
diffs_ftn = np.abs


class PhaseAnnealingAlgObjective:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _get_obj_scorr_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):

                    sim_diffs = self._sim_scorr_diffs[(label, lag)]

                    ftn = self._ref_scorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                    obj_val += sq_diffs.sum() * self._sett_wts_lag_wts[i]

        else:
            # TODO: Add weights here to all as well
            obj_val = ((self._ref_scorrs - self._sim_scorrs) ** 2).sum()

        return obj_val

    def _get_obj_asymms_1_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):
                    sim_diffs = self._sim_asymm_1_diffs[(label, lag)]

                    ftn = self._ref_asymm_1_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    if self._alg_cdf_opt_idxs_flag:
                        if (label, lag) not in self._alg_cdf_opt_asymms_1_sims:
                            self._alg_cdf_opt_asymms_1_sims[(label, lag)] = []

                        self._alg_cdf_opt_asymms_1_sims[(label, lag)].append(sim_probs)
                        continue

                    if self._alg_asymm_1_dist_sclr != 1:
                        sim_diffs_shft = sim_diffs.copy()

                        diffs = ref_probs - sim_probs

                        # Scaling everywhere.
                        sim_diffs_shft[(diffs > 0) & (ref_probs <= 0.5)] *= self._alg_asymm_1_dist_sclr
                        sim_diffs_shft[(diffs > 0) & (ref_probs > 0.5)] *= (1 / self._alg_asymm_1_dist_sclr)

                        sim_diffs_shft[(diffs <= 0) & (ref_probs <= 0.5)] *= (1 / self._alg_asymm_1_dist_sclr)
                        sim_diffs_shft[(diffs <= 0) & (ref_probs > 0.5)] *= self._alg_asymm_1_dist_sclr

                        # Partial scaling.
#                         sim_diffs_shft[(diffs <= 0) & (ref_probs <= 0.5)] *= (1 / self._alg_asymm_1_dist_sclr)
#                         sim_diffs_shft[~((diffs <= 0) & (ref_probs <= 0.5))] *= self._alg_asymm_1_dist_sclr

                        sim_probs_shft = ftn(sim_diffs_shft)

                        # Plain scaling.
#                         sim_probs_shft = ftn(sim_diffs * self._alg_asymm_1_dist_sclr)

                    else:
                        sim_probs_shft = sim_probs

                    sq_diffs = diffs_ftn((ref_probs - sim_probs_shft) * ftn.wts) ** diffs_exp

                    if self._alg_cdf_opt_asymms_1_idxs is not None:
                        sq_diffs *= self._alg_cdf_opt_asymms_1_idxs[
                            (label, lag)]

                    obj_val += sq_diffs.sum() * self._sett_wts_lag_wts[i]

                    if self._alg_done_opt_flag:
                        import matplotlib.pyplot as plt
                        plt.ioff()
                        plt.style.use('ggplot')
                        plt.plot(ftn.y, ftn.y, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, sim_probs, c='blue', alpha=0.7, lw=2)
                        plt.plot(ftn.y, ftn.ks_u_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, ftn.ks_l_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, sim_probs_shft, c='red', alpha=0.7, lw=1)
                        plt.grid()
                        plt.title(f'Asymm 1, Lag: {lag}')
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        plt.show()
                        plt.close()

        else:
            obj_val = ((self._ref_asymms_1 - self._sim_asymms_1) ** 2).sum()

        return obj_val

    def _get_obj_asymms_2_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):
                    sim_diffs = self._sim_asymm_2_diffs[(label, lag)].copy()

                    ftn = self._ref_asymm_2_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    if self._alg_cdf_opt_idxs_flag:
                        if (label, lag) not in self._alg_cdf_opt_asymms_2_sims:
                            self._alg_cdf_opt_asymms_2_sims[(label, lag)] = []

                        self._alg_cdf_opt_asymms_2_sims[(label, lag)].append(sim_probs)
                        continue

                    if self._alg_asymm_2_dist_sclr != 1:
                        sim_diffs_shft = sim_diffs.copy()

                        diffs = ref_probs - sim_probs

                        # Scaling everywhere.
                        sim_diffs_shft[(diffs > 0) & (ref_probs <= 0.5)] *= self._alg_asymm_2_dist_sclr
                        sim_diffs_shft[(diffs > 0) & (ref_probs > 0.5)] *= (1 / self._alg_asymm_2_dist_sclr)

                        sim_diffs_shft[(diffs <= 0) & (ref_probs <= 0.5)] *= (1 / self._alg_asymm_2_dist_sclr)
                        sim_diffs_shft[(diffs <= 0) & (ref_probs > 0.5)] *= self._alg_asymm_2_dist_sclr

                        # Partial scaling.
#                         sim_diffs_shft[(diffs <= 0) & (ref_probs <= 0.5)] *= (1 / self._alg_asymm_2_dist_sclr)
#                         sim_diffs_shft[~((diffs <= 0) & (ref_probs <= 0.5))] *= self._alg_asymm_2_dist_sclr

                        sim_probs_shft = ftn(sim_diffs_shft)

                        # Plain scaling.
#                         sim_probs_shft = ftn(sim_diffs * self._alg_asymm_2_dist_sclr)

                    else:
                        sim_probs_shft = sim_probs

                    sq_diffs = diffs_ftn((ref_probs - sim_probs_shft) * ftn.wts) ** diffs_exp

                    if self._alg_cdf_opt_asymms_2_idxs is not None:
                        sq_diffs *= self._alg_cdf_opt_asymms_2_idxs[
                            (label, lag)]

                    obj_val += sq_diffs.sum() * self._sett_wts_lag_wts[i]

                    if self._alg_done_opt_flag:
                        import matplotlib.pyplot as plt
                        plt.ioff()
                        plt.style.use('ggplot')
                        plt.plot(ftn.y, ftn.y, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, sim_probs, c='blue', alpha=0.7, lw=2)
                        plt.plot(ftn.y, ftn.ks_u_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, ftn.ks_l_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, sim_probs_shft, c='red', alpha=0.7, lw=1)
                        plt.grid()
                        plt.title(f'Asymm 2, Lag: {lag}')
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        plt.show(block=True)
                        plt.close()

        else:
            obj_val = ((self._ref_asymms_2 - self._sim_asymms_2) ** 2).sum()

        return obj_val

    def _get_obj_ecop_dens_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):

                    sim_diffs = self._sim_ecop_dens_diffs[(label, lag)]

                    ftn = self._ref_ecop_dens_diffs_cdfs_dict[(label, lag)]

                    sim_probs = ftn(sim_diffs)

                    ref_probs = ftn.y

                    sq_diff = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                    obj_val += (
                        sq_diff.sum() / ftn.sclr) * self._sett_wts_lag_wts[i]

        else:
            obj_val = (
                (self._ref_ecop_dens -
                 self._sim_ecop_dens) ** 2).sum()

        return obj_val

    def _get_obj_ecop_etpy_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):
                    sim_diffs = self._sim_ecop_etpy_diffs[(label, lag)]

                    ftn = self._ref_ecop_etpy_diffs_cdfs_dict[(label, lag)]

                    sim_probs = ftn(sim_diffs)

                    ref_probs = ftn.y

                    sq_diff = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                    obj_val += (
                        (sq_diff.sum() / ftn.sclr) * self._sett_wts_lag_wts[i])

        else:
            obj_val = ((self._ref_ecop_etpy - self._sim_ecop_etpy) ** 2).sum()

        return obj_val

    def _get_obj_nth_ord_diffs_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, nth_ord in enumerate(self._sett_obj_nth_ords):

                    sim_diffs = self._sim_nth_ord_diffs[
                        (label, nth_ord)].copy()

                    ftn = self._ref_nth_ord_diffs_cdfs_dict[(label, nth_ord)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                    obj_val += sq_diffs.sum() * self._sett_wts_nth_wts[i]

                    if self._alg_done_opt_flag:
                        import matplotlib.pyplot as plt
                        plt.ioff()
                        plt.style.use('ggplot')
                        plt.plot(ftn.y, ftn.y, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, sim_probs, c='blue', alpha=0.7, lw=2)
                        plt.plot(ftn.y, ftn.ks_u_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.plot(ftn.y, ftn.ks_l_bds, c='grey', alpha=0.7, lw=1, ls='--')
                        plt.grid()
                        plt.title(f'Nth: {nth_ord}')
                        mng = plt.get_current_fig_manager()
                        mng.window.state('zoomed')
                        plt.show(block=True)
                        plt.close()

        else:
            obj_val = ((self._ref_nths - self._sim_nths) ** 2).sum()

        return obj_val

    def _get_obj_cos_sin_dist_val(self):

        obj_val = 0.0
        for i, label in enumerate(self._data_ref_labels):
            cos_ftn = self._ref_cos_sin_cdfs_dict[(label, 'cos')]
            ref_probs_cos = cos_ftn.y
            sim_probs_cos = np.sort(cos_ftn(self._sim_ft.real[:, i]))
            cos_sq_diffs = diffs_ftn((ref_probs_cos - sim_probs_cos) * cos_ftn.wts) ** diffs_exp
            obj_val += cos_sq_diffs.sum() / cos_ftn.sclr

            sin_ftn = self._ref_cos_sin_cdfs_dict[(label, 'sin')]
            ref_probs_sin = sin_ftn.y
            sim_probs_sin = np.sort(sin_ftn(self._sim_ft.imag[:, i]))
            sin_sq_diffs = diffs_ftn((ref_probs_sin - sim_probs_sin) * sin_ftn.wts) ** diffs_exp
            obj_val += sin_sq_diffs.sum() / sin_ftn.sclr

        return obj_val

    def _get_obj_pcorr_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for i, lag in enumerate(self._sett_obj_lag_steps):

                    sim_diffs = self._sim_pcorr_diffs[(label, lag)]

                    ftn = self._ref_pcorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                    obj_val += sq_diffs.sum() * self._sett_wts_lag_wts[i]

        else:
            obj_val = ((self._ref_pcorrs - self._sim_pcorrs) ** 2).sum()

        return obj_val

    def _get_obj_asymms_1_ms_val(self):

        obj_val = 0.0
        if self._sett_obj_use_obj_dist_flag:
            for comb in self._ref_mult_asymm_1_diffs_cdfs_dict:
                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                sim_diffs = self._sim_mult_asymms_1_diffs[comb]

                ftn = self._ref_mult_asymm_1_diffs_cdfs_dict[comb]

                ref_probs = ftn.y

                sim_probs = ftn(sim_diffs)

                sq_diffs = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                obj_val += sq_diffs.sum()

        else:
            for comb in self._ref_mult_asymm_1_diffs_cdfs_dict:
                ref_diffs = (
                    self._ref_mult_asymm_1_diffs_cdfs_dict[comb].x.sum())

                sim_diffs = self._sim_mult_asymms_1_diffs[comb].sum()

                obj_val += ((ref_diffs - sim_diffs) ** 2).sum()

        return obj_val

    def _get_obj_asymms_2_ms_val(self):

        obj_val = 0.0
        if self._sett_obj_use_obj_dist_flag:
            for comb in self._ref_mult_asymm_2_diffs_cdfs_dict:
                if len(comb) != 2:
                    raise NotImplementedError(
                        'Asymmetry 2 configured for pairs only!')

                sim_diffs = self._sim_mult_asymms_2_diffs[comb]

                ftn = self._ref_mult_asymm_2_diffs_cdfs_dict[comb]

                ref_probs = ftn.y

                sim_probs = ftn(sim_diffs)

                sq_diffs = diffs_ftn((ref_probs - sim_probs) * ftn.wts) ** diffs_exp

                obj_val += sq_diffs.sum()

        else:
            for comb in self._ref_mult_asymm_2_diffs_cdfs_dict:
                ref_diffs = (
                    self._ref_mult_asymm_2_diffs_cdfs_dict[comb].x.sum())

                sim_diffs = self._sim_mult_asymms_2_diffs[comb].sum()

                obj_val += ((ref_diffs - sim_diffs) ** 2).sum()

        return obj_val

    def _get_obj_ecop_dens_ms_val(self):

        obj_val = 0.0
        for comb in self._ref_mult_ecop_dens_diffs_cdfs_dict:
            obj_val += ((
                self._ref_mult_ecop_dens_diffs_cdfs_dict[comb] -
                self._sim_mult_ecops_dens_diffs[comb]) ** 2).sum()

        return obj_val

    def _get_obj_data_ft_val(self):

        obj_val = (
            ((self._ref_data_ft - self._sim_data_ft)) ** 2).sum()

        return obj_val

    @PAP._timer_wrap
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

        obj_vals = np.array(obj_vals, dtype=np.float64)

        assert np.all(np.isfinite(obj_vals)), 'Invalid obj_vals!'

        if self._sett_wts_obj_wts is not None:
            obj_vals *= self._sett_wts_obj_wts

        return obj_vals


class PhaseAnnealingAlgIO:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _write_cls_rltzn(self, rltzn_iter, ret):

        with self._lock:
            h5_path = self._sett_misc_outs_dir / self._save_h5_name

            with h5py.File(h5_path, mode='a', driver=None) as h5_hdl:
                self._write_ref_cls_rltzn(h5_hdl)
                self._write_sim_cls_rltzn(h5_hdl, rltzn_iter, ret)
        return

    def _write_ref_cls_rltzn(self, h5_hdl):

        # Should be called by _write_cls_rltzn with a lock

        cls_pad_zeros = len(str(self._ref_phs_ann_class_vars[2]))

        ref_cls_grp_lab = (
            f'data_ref_rltzn/'
            f'{self._ref_phs_ann_class_vars[3]:0{cls_pad_zeros}d}')

        if ref_cls_grp_lab in h5_hdl:
            return

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_ref_*'):
                continue

            datas.append((var, getattr(self, var)))

        ref_cls_grp = h5_hdl.create_group(ref_cls_grp_lab)

        ll_idx = 0  # ll is for label
        lg_idx = 1  # lg is for lag

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                ref_cls_grp[data_lab] = data_val

            elif isinstance(data_val, interp1d):
                ref_cls_grp[data_lab + '_x'] = data_val.x
                ref_cls_grp[data_lab + '_y'] = data_val.y

            elif (isinstance(data_val, dict) and

                  all([isinstance(key[lg_idx], np.int64)
                       for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'
                    ref_cls_grp[data_lab + f'{lab}_x'] = data_val[key].x
                    ref_cls_grp[data_lab + f'{lab}_y'] = data_val[key].y

            elif (isinstance(data_val, dict) and

                  all([key[lg_idx] in ('cos', 'sin') for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]}'
                    ref_cls_grp[data_lab + f'{lab}_x'] = data_val[key].x
                    ref_cls_grp[data_lab + f'{lab}_y'] = data_val[key].y

            elif (isinstance(data_val, dict) and

                  all([isinstance(key[lg_idx], np.int64)
                       for key in data_val]) and

                  all([isinstance(val, np.ndarray)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'
                    ref_cls_grp[data_lab + lab] = data_val[key]

            elif (isinstance(data_val, dict) and

                  all([all([col in self._data_ref_labels for col in key])
                       for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    comb_str = '_'.join(key)
                    ref_cls_grp[f'{data_lab}_{comb_str}_x'] = data_val[key].x
                    ref_cls_grp[f'{data_lab}_{comb_str}_y'] = data_val[key].y

            elif isinstance(data_val, (str, float, int)):
                ref_cls_grp.attrs[data_lab] = data_val

            elif data_val is None:
                ref_cls_grp.attrs[data_lab] = str(data_val)

            elif (isinstance(data_val, dict) and

                 all([isinstance(data_val[key], np.ndarray)
                      for key in data_val])):

                pass

            else:
                raise NotImplementedError(
                    f'Unknown type {type(data_val)} for variable '
                    f'{data_lab}!')

        h5_hdl.flush()
        return

    def _write_sim_cls_rltzn(self, h5_hdl, rltzn_iter, ret):

        # Should be called by _write_cls_rltzn with a lock

        sim_pad_zeros = len(str(self._sett_misc_n_rltzns))
        cls_pad_zeros = len(str(self._sim_phs_ann_class_vars[2]))

        main_sim_grp_lab = 'data_sim_rltzns'

        sim_grp_lab = f'{rltzn_iter:0{sim_pad_zeros}d}'

        sim_cls_grp_lab = (
            f'{self._sim_phs_ann_class_vars[3]:0{cls_pad_zeros}d}')

        if not main_sim_grp_lab in h5_hdl:
            sim_grp_main = h5_hdl.create_group(main_sim_grp_lab)

        else:
            sim_grp_main = h5_hdl[main_sim_grp_lab]

        if not sim_grp_lab in sim_grp_main:
            sim_grp = sim_grp_main.create_group(sim_grp_lab)

        else:
            sim_grp = sim_grp_main[sim_grp_lab]

        if not sim_cls_grp_lab in sim_grp:
            sim_cls_grp = sim_grp.create_group(sim_cls_grp_lab)

        else:
            sim_cls_grp = sim_grp[sim_cls_grp_lab]

        for lab, val in ret._asdict().items():
            if isinstance(val, np.ndarray):
                sim_cls_grp[lab] = val

            elif fnmatch(lab, 'tmr*') and isinstance(val, dict):
                tmr_grp = sim_cls_grp.create_group(lab)
                for meth_name, meth_val in val.items():
                    tmr_grp.attrs[meth_name] = meth_val

            else:
                sim_cls_grp.attrs[lab] = val

        if self._sim_mag_spec_flags is not None:
            sim_cls_grp['sim_mag_spec_flags'] = (
                self._sim_mag_spec_flags)

        if self._sim_mag_spec_idxs is not None:
            sim_cls_grp['sim_mag_spec_idxs'] = self._sim_mag_spec_idxs

        h5_hdl.flush()
        return


class PhaseAnnealingAlgRealization:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _show_rltzn_situ(self, iter_ctr, rltzn_iter):

        c1 = self._sett_ann_max_iters >= 10000
        c2 = not (iter_ctr % (0.1 * self._sett_ann_max_iters))

        if c1 and c2:
            with self._lock:
                print_sl()

                print(
                    f'Realization {rltzn_iter} finished {iter_ctr} out of '
                    f'{self._sett_ann_max_iters} iterations at {asctime()}.')

                print_el()
        return

    def _update_cdf_sclrs(self, iter_ctr):

        if self._alg_asymm_1_dist_sclr != 1:
            self._alg_asymm_1_dist_sclr += (
                self._alg_asymm_1_dist_sclr_slp)

            self._alg_asymm_1_dist_sclr = max(
                self._alg_asymm_1_dist_sclr, 1.0)

            if self._alg_asymm_1_dist_sclr > 1.0:
                self._alg_force_acpt_flag = True

                if (self._sett_misc_n_rltzns == 1) and self._vb:
                    print(
                        'iter_ctr, alg_asymm_1_dist_sclr:',
                        iter_ctr,
                        self._alg_asymm_1_dist_sclr)

        if self._alg_asymm_2_dist_sclr != 1:
            self._alg_asymm_2_dist_sclr += (
                self._alg_asymm_2_dist_sclr_slp)

            self._alg_asymm_2_dist_sclr = max(
                self._alg_asymm_2_dist_sclr, 1.0)

            if self._alg_asymm_2_dist_sclr > 1.0:
                self._alg_force_acpt_flag = True

                if (self._sett_misc_n_rltzns == 1) and self._vb:
                    print(
                        'iter_ctr, alg_asymm_2_dist_sclr:',
                        iter_ctr,
                        self._alg_asymm_2_dist_sclr)

        return

    def _update_obj_wts(self, obj_vals_all_indiv, iter_ctr):

        '''
        Called during auto_temp_init or temp_updt.
        '''

        if not self._sett_wts_obj_auto_set_flag:
            return

        c0 = self._alg_ann_runn_auto_init_temp_search_flag
        c1 = not self._sett_auto_temp_set_flag
        c2 = iter_ctr == self._sett_wts_obj_init_iter
        c3 = self._sett_wts_obj_updt_with_temp_flag
        c4 = iter_ctr >= self._sett_wts_obj_init_iter

        ca = c0 and c2
        cb = (not c0) and ((c1 and c2) or (c3 and c4))

        if ca or cb:
            subset = obj_vals_all_indiv[-self._sett_wts_obj_take_mean_iters:]

            # Means should be pure obj_vals i.e. without the wts.
            # That's why scaling them with inverse of wts.
            # It would be better to use actual values instead of scaled ones.
            means = np.array(subset).mean(axis=0)

            if self._sett_wts_obj_wts is not None:
                means /= self._sett_wts_obj_wts

            sum_means = means.sum()

            obj_wts = []
            for i in range(means.size):
                obj_wt = sum_means / means[i]
                obj_wts.append(obj_wt)

            obj_wts = np.array(obj_wts)
            self._sett_wts_obj_wts = (obj_wts.size * obj_wts) / obj_wts.sum()

            if self._vb and ca:
                print(ca, cb, iter_ctr, self._sett_wts_obj_wts)

            self._alg_force_acpt_flag = True

        return

    def _get_stopp_criteria(self, test_vars):

        (iter_ctr,
         iters_wo_acpt,
         tol,
         temp,
         phs_red_rate,
         acpt_rate) = test_vars

        stopp_criteria = (
            (iter_ctr < self._sett_ann_max_iters),
            (iters_wo_acpt < self._sett_ann_max_iter_wo_chng),
            (tol > self._sett_ann_obj_tol),
#             (not np.isclose(temp, 0.0)),
            (temp > 1e-15),
#             (not np.isclose(phs_red_rate, 0.0)),
            (phs_red_rate > 1e-15),
            (acpt_rate > self._sett_ann_stop_acpt_rate),
            )

        return stopp_criteria

    def _get_phs_red_rate(self, iter_ctr, acpt_rate, old_phs_red_rate):

        if self._alg_ann_runn_auto_init_temp_search_flag:
            phs_red_rate = 1.0

        else:
            if self._sett_ann_phs_red_rate_type == 0:
                phs_red_rate = 1.0

            elif self._sett_ann_phs_red_rate_type == 1:
                phs_red_rate = 1.0 - (iter_ctr / self._sett_ann_max_iters)

            elif self._sett_ann_phs_red_rate_type == 2:
                phs_red_rate = float((
                    self._sett_ann_phs_red_rate **
                    (iter_ctr // self._sett_ann_upt_evry_iter)))

            elif self._sett_ann_phs_red_rate_type == 3:

                # An unstable mean of acpts_rjts_dfrntl is a
                # problem. So, it has to be long enough.
                phs_red_rate = min(acpt_rate, old_phs_red_rate)

            else:
                raise NotImplemented(
                    'Unknown _sett_ann_phs_red_rate_type:',
                    self._sett_ann_phs_red_rate_type)

            assert phs_red_rate >= 0.0, 'Invalid phs_red_rate!'

        return phs_red_rate

    def _get_phs_idxs_sclr(self, iter_ctr, acpt_rate, old_idxs_sclr):

        if not self._sett_mult_phs_flag:
            idxs_sclr = np.nan

        else:
            if self._sett_mult_phs_sample_type == 0:
                idxs_sclr = 1.0

            elif self._sett_mult_phs_sample_type == 1:
                idxs_sclr = 1.0 - (iter_ctr / self._sett_ann_max_iters)

            elif self._sett_mult_phs_sample_type == 2:
                idxs_sclr = float((
                    self._sett_mult_phss_red_rate **
                    (iter_ctr // self._sett_ann_upt_evry_iter)))

            elif self._sett_mult_phs_sample_type == 3:
                idxs_sclr = min(acpt_rate, old_idxs_sclr)

            else:
                raise NotImplementedError

        return idxs_sclr

    def _get_next_idxs(self, idxs_sclr):

        # _sim_mag_spec_cdf makes it difficult without a while-loop.

        min_idx = self._sim_phs_ann_class_vars[0]

        idxs_diff = (
            self._sim_phs_ann_class_vars[1] - self._sim_phs_ann_class_vars[0])

        assert idxs_diff > 0, idxs_diff

        if self._sett_mult_phs_flag:
            min_idx_to_gen = self._sett_mult_phs_n_beg_phss
            max_idxs_to_gen = self._sett_mult_phs_n_end_phss

        else:
            min_idx_to_gen = 1
            max_idxs_to_gen = 2

        # Inclusive
        min_idxs_to_gen = min([min_idx_to_gen, idxs_diff])

        # Inclusive
        max_idxs_to_gen = min([max_idxs_to_gen, idxs_diff])

        if np.isnan(idxs_sclr):
            idxs_to_gen = np.random.randint(
                min_idxs_to_gen, max_idxs_to_gen)

        else:
            idxs_to_gen = min_idxs_to_gen + (
                int(round(idxs_sclr * (max_idxs_to_gen - min_idxs_to_gen))))

        max_ctr = 100 * self._sim_shape[0] * self._data_ref_n_labels

#         print(idxs_to_gen)

        assert min_idx_to_gen >= 1, 'This shouldn\'t have happend!'
        assert idxs_to_gen >= 1, 'This shouldn\'t have happend!'

        if min_idx_to_gen == idxs_diff:
            new_idxs = np.arange(1, min_idxs_to_gen + 1)

        else:
            new_idxs = []

        while len(new_idxs) < idxs_to_gen:

            idx_ctr = 0
            while True:

                if self._sett_ann_mag_spec_cdf_idxs_flag:
                    index = int(self._sim_mag_spec_cdf(np.random.random()))

                else:
                    index = min_idx + int(np.random.random() * idxs_diff)

                assert 0 < index < self._sim_shape[0], (
                    f'Invalid index {index}!')

                idx_ctr += 1

                if idx_ctr == max_ctr:
                    assert RuntimeError('Could not find a suitable index!')

                if index in new_idxs:
                    continue

                if (self._sim_phs_ann_class_vars[0] <=
                    index <=
                    self._sim_phs_ann_class_vars[1]):

                    new_idxs.append(index)

                    break

                else:
                    continue

        return np.array(new_idxs, dtype=int)

    @PAP._timer_wrap
    def _get_next_iter_vars(self, phs_red_rate, idxs_sclr):

        new_idxs = self._get_next_idxs(idxs_sclr)

        # Making a copy of the phases is important if not then the
        # returned old_phs and new_phs are SOMEHOW the same.
        old_phss = self._sim_phs_spec[new_idxs, :].copy()

        new_phss = -np.pi + (
            2 * np.pi * np.random.random((old_phss.shape[0], 1)))

        if self._alg_ann_runn_auto_init_temp_search_flag:
            pass

        else:
            new_phss *= phs_red_rate

        new_phss = old_phss + new_phss

        new_rect_phss = np.full(new_phss.shape, np.nan)

        for i in range(new_phss.shape[0]):
            for j in range(new_phss.shape[1]):

                # Didn't work without copy.
                new_phs = new_phss[i, j].copy()

                if new_phs > +np.pi:
                    ratio = (new_phs / +np.pi) - 1
                    new_phs = -np.pi * (1 - ratio)

                elif new_phs < -np.pi:
                    ratio = (new_phs / -np.pi) - 1
                    new_phs = +np.pi * (1 - ratio)

                assert (-np.pi <= new_phs <= +np.pi)

                new_rect_phss[i, j] = new_phs

        assert np.all(np.isfinite(new_rect_phss)), 'Invalid phases!'

        new_phss = new_rect_phss

        old_coeff = None
        new_coeff = None

        if (self._sett_extnd_len_set_flag
            and np.any(self._sim_mag_spec_flags[new_idxs])):

            raise NotImplementedError('Not for mult cols yet!')

#             rand = (expon.ppf(
#                 np.random.random(),
#                 scale=self._ref_mag_spec_mean * self._sett_extnd_len_rel_shp[0])
#                 ) * phs_red_rate

            # FIXME: There should be some sort scaling of this.
            # Convergence could become really slow if magnitudes are large
            # or too fast if magniudes are small comparatively.
            # Scaling could be based on reference magnitudes.
#             rand = (-1 + (2 * np.random.random())) * phs_red_rate
#
#             old_coeff = self._sim_ft[new_idxs].copy()
#
#             old_mag = np.abs(old_coeff)
#
#             rand += old_mag
#
#             rand = max(0, rand)
#
#             new_coeff_incr = (
#                 (rand * np.cos(new_phs)) + (rand * np.sin(new_phs) * 1j))
#
#             new_coeff = new_coeff_incr

        return old_phss, new_phss, old_coeff, new_coeff, new_idxs

    def _update_sim_no_prms(self):

        data = self._get_non_mono_data(np.fft.irfft(self._sim_ft, axis=0))

        probs = self._get_probs(data, True)

        self._sim_data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._sim_data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._sim_probs = probs

        self._update_obj_vars('sim')
        return

    @PAP._timer_wrap
    def _update_sim(self, idxs, phss, coeffs):

        self._sim_phs_spec[idxs] = phss

        if coeffs is not None:
            self._sim_ft[idxs] = coeffs
            self._sim_mag_spec[idxs] = np.abs(self._sim_ft[idxs])

        else:
            self._sim_ft.real[idxs] = np.cos(phss) * self._sim_mag_spec[idxs]
            self._sim_ft.imag[idxs] = np.sin(phss) * self._sim_mag_spec[idxs]

        self._update_sim_no_prms()
        return

    def _gen_gnrc_rltzn(self, args):

        (rltzn_iter,
         pre_init_temps,
         pre_acpt_rates,
         init_temp) = args

        beg_time = default_timer()

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            assert 0 <= rltzn_iter < self._sett_ann_auto_init_temp_atpts, (
                    'Invalid rltzn_iter!')

            if (self._sett_wts_obj_auto_set_flag and
                (self._sett_wts_obj_wts is not None)):

                self._sett_wts_obj_wts = None

        else:
            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                    'Invalid rltzn_iter!')

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implemention for 2D only!')

        # Randomize all phases before starting.
        self._gen_sim_aux_data()

        # Initialize sim anneal variables.
        iter_ctr = 0
        acpt_rate = 1.0

        temp = self._get_init_temp(
            rltzn_iter, pre_init_temps, pre_acpt_rates, init_temp)

        phs_red_rate = self._get_phs_red_rate(iter_ctr, acpt_rate, 1.0)

        idxs_sclr = self._get_phs_idxs_sclr(iter_ctr, acpt_rate, 1.0)

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            iters_wo_acpt = 0
            tol = np.inf

            tols_dfrntl = deque(maxlen=self._sett_ann_obj_tol_iters)

            acpts_rjts_dfrntl = deque(
                maxlen=self._sett_ann_acpt_rate_iters)

            stopp_criteria = self._get_stopp_criteria(
                (iter_ctr,
                 iters_wo_acpt,
                 tol,
                 temp,
                 phs_red_rate,
                 acpt_rate))

        old_idxs = self._get_next_idxs(idxs_sclr)
        new_idxs = old_idxs

        old_obj_val = self._get_obj_ftn_val().sum()

        # Initialize diagnostic variables.
        acpts_rjts_all = []

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            tols = []

            obj_vals_all = []

            obj_val_min = np.inf
            obj_vals_min = []

            phs_red_rates = [[iter_ctr, phs_red_rate]]

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

            idxs_sclrs = [[iter_ctr, idxs_sclr]]

        else:
            pass

        obj_vals_all_indiv = []

        while all(stopp_criteria):

            #==============================================================
            # Simulated annealing start.
            #==============================================================

            (old_phss,
             new_phss,
             old_coeffs,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs)

            new_obj_val_indiv = self._get_obj_ftn_val()
            new_obj_val = new_obj_val_indiv.sum()

            old_new_diff = old_obj_val - new_obj_val

            if old_new_diff > 0:
                accept_flag = True

            else:
                rand_p = np.random.random()

                boltz_p = np.exp(old_new_diff / temp)

                if rand_p < boltz_p:
                    accept_flag = True

                else:
                    accept_flag = False

            if self._alg_force_acpt_flag:
                accept_flag = True
                self._alg_force_acpt_flag = False

            if accept_flag:
                old_idxs = new_idxs

                old_obj_val = new_obj_val

            else:
                self._update_sim(new_idxs, old_phss, old_coeffs)

            iter_ctr += 1

            #==============================================================
            # Simulated annealing end.
            #==============================================================

            acpts_rjts_all.append(accept_flag)

            obj_vals_all_indiv.append(new_obj_val_indiv)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                stopp_criteria = (
                    (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                    )

                self._update_obj_wts(obj_vals_all_indiv, iter_ctr)

            else:
                if new_obj_val < obj_val_min:
                    self._sim_ft_best = self._sim_ft.copy()

                tols_dfrntl.append(abs(old_new_diff))

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    obj_val_min = min(obj_val_min, new_obj_val)

                obj_vals_min.append(obj_val_min)
                obj_vals_all.append(new_obj_val)

                acpts_rjts_dfrntl.append(accept_flag)

                self._sim_n_idxs_all_cts[new_idxs] += 1

                if iter_ctr >= tols_dfrntl.maxlen:
                    tol = sum(tols_dfrntl) / float(tols_dfrntl.maxlen)

                    assert np.isfinite(tol), 'Invalid tol!'

                    tols.append(tol)

                if accept_flag:
                    self._sim_n_idxs_acpt_cts[new_idxs] += 1
                    iters_wo_acpt = 0

                else:
                    iters_wo_acpt += 1

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    acpt_rates_dfrntl.append([iter_ctr - 1, acpt_rate])

                    acpt_rate = (
                        sum(acpts_rjts_dfrntl) /
                        float(acpts_rjts_dfrntl.maxlen))

                    acpt_rates_dfrntl.append([iter_ctr, acpt_rate])

                if (iter_ctr % self._sett_ann_upt_evry_iter) == 0:

                    # Temperature
                    temps.append([iter_ctr - 1, temp])

                    temp *= self._sett_ann_temp_red_rate

                    assert temp >= 0.0, 'Invalid temp!'

                    temps.append([iter_ctr, temp])

                    # Phase reduction rate
                    phs_red_rates.append([iter_ctr - 1, phs_red_rate])

                    phs_red_rate = self._get_phs_red_rate(
                        iter_ctr, acpt_rate, phs_red_rate)

                    phs_red_rates.append([iter_ctr, phs_red_rate])

                    # Phase indices reduction rate
                    idxs_sclrs.append([iter_ctr - 1, idxs_sclr])

                    idxs_sclr = self._get_phs_idxs_sclr(
                        iter_ctr, acpt_rate, idxs_sclr)

                    idxs_sclrs.append([iter_ctr, idxs_sclr])

#                     if phs_red_rate < 0.05:
#                         self._update_cdf_sclrs(iter_ctr)

                    # Objective function weights
                    self._update_obj_wts(obj_vals_all_indiv, iter_ctr)

                if self._vb:
                    self._show_rltzn_situ(iter_ctr, rltzn_iter)

                stopp_criteria = self._get_stopp_criteria(
                    (iter_ctr,
                     iters_wo_acpt,
                     tol,
                     temp,
                     phs_red_rate,
                     acpt_rate))

        # Manual update of timer because this function writes timings
        # to the HDF5 file before it returns.
        if '_gen_gnrc_rltzns' not in self._sim_tmr_cumm_call_times:
            self._sim_tmr_cumm_call_times['_gen_gnrc_rltzns'] = 0.0
            self._sim_tmr_cumm_n_calls['_gen_gnrc_rltzns'] = 0.0

        self._sim_tmr_cumm_call_times['_gen_gnrc_rltzns'] += (
            default_timer() - beg_time)

        self._sim_tmr_cumm_n_calls['_gen_gnrc_rltzns'] += 1

        if self._alg_ann_runn_auto_init_temp_search_flag:
            if self._sett_wts_obj_auto_set_flag:
                ret_idx = self._sett_wts_obj_init_iter + 1

            else:
                ret_idx = 0

            ret = (
                sum(acpts_rjts_all[ret_idx:]) / len(acpts_rjts_all[ret_idx:]),
                temp)

        else:
            assert self._sim_n_idxs_all_cts[+0] == 0
            assert self._sim_n_idxs_all_cts[-1] == 0

#             if self._sett_misc_n_rltzns == 1:
#                 self._alg_done_opt_flag = True
#
#                 self._get_obj_ftn_val()
#
#                 self._alg_done_opt_flag = False
#
#             print('alg_asymm_2_dist_sclr:', self._alg_asymm_2_dist_sclr)
#             print('alg_asymm_1_dist_sclr:', self._alg_asymm_1_dist_sclr)

            self._update_ref_at_end()
            self._update_sim_at_end()

            acpts_rjts_all = np.array(acpts_rjts_all, dtype=bool)

            acpt_rates_all = (
                np.cumsum(acpts_rjts_all) /
                np.arange(1, acpts_rjts_all.size + 1, dtype=float))

            if ((not self._sett_extnd_len_set_flag) or
                (self._sett_extnd_len_rel_shp[0] == 1)):

                ref_sim_ft_corr = self._get_cumm_ft_corr(
                        self._ref_ft, self._sim_ft).astype(np.float64)

            else:
                ref_sim_ft_corr = np.array([], dtype=np.float64)

            sim_sim_ft_corr = self._get_cumm_ft_corr(
                    self._sim_ft, self._sim_ft).astype(np.float64)

            out_data = [
                self._sim_ft,
                self._sim_mag_spec,
                self._sim_phs_spec,
                self._sim_probs,
                self._sim_scorrs,
                self._sim_asymms_1,
                self._sim_asymms_2,
                self._sim_ecop_dens,
                self._sim_ecop_etpy,
                self._sim_data_ft,
                iter_ctr,
                iters_wo_acpt,
                tol,
                temp,
                np.array(stopp_criteria),
                np.array(tols, dtype=np.float64),
                np.array(obj_vals_all, dtype=np.float64),
                acpts_rjts_all,
                acpt_rates_all,
                np.array(obj_vals_min, dtype=np.float64),
                np.array(temps, dtype=np.float64),
                np.array(phs_red_rates, dtype=np.float64),
                self._sim_n_idxs_all_cts,
                self._sim_n_idxs_acpt_cts,
                np.array(acpt_rates_dfrntl, dtype=np.float64),
                ref_sim_ft_corr,
                sim_sim_ft_corr,
                self._sim_phs_ann_class_vars,
                self._sim_data,
                self._sim_pcorrs,
                self._sim_phs_mod_flags,
                np.array(obj_vals_all_indiv, dtype=np.float64),
                self._sim_nths,
                np.array(idxs_sclrs, dtype=np.float64),
                self._sim_tmr_cumm_call_times,
                self._sim_tmr_cumm_n_calls,
                ]

            out_data.extend(
                [val for val in self._sim_nth_ord_diffs.values()])

            out_data.extend(
                [val for val in self._sim_scorr_diffs.values()])

            out_data.extend(
                [val for val in self._sim_asymm_1_diffs.values()])

            out_data.extend(
                [val for val in self._sim_asymm_2_diffs.values()])

            out_data.extend(
                [val for val in self._sim_ecop_dens_diffs.values()])

            out_data.extend(
                [val for val in self._sim_ecop_etpy_diffs.values()])

            out_data.extend(
                [val for val in self._sim_pcorr_diffs.values()])

            if self._data_ref_n_labels > 1:
                out_data.extend(
                    [val for val in self._sim_mult_asymms_1_diffs.values()])

                out_data.extend(
                    [val for val in self._sim_mult_asymms_2_diffs.values()])

#                 out_data.extend(
#                     [val for val in self._sim_mult_ecops_dens_diffs.values()])

            self._write_cls_rltzn(
                rltzn_iter, self._sim_rltzns_proto_tup._make(out_data))

            ret = stopp_criteria

        return ret

    def _gen_gnrc_rltzns(self, args):

        ((rltzn_iter_beg, rltzn_iter_end),
        ) = args

        rltzns = []
        pre_init_temps = []
        pre_acpt_rates = []

        for rltzn_iter in range(rltzn_iter_beg, rltzn_iter_end):
            rltzn_args = (
                rltzn_iter,
                pre_init_temps,
                pre_acpt_rates,
                None,
                )

            rltzn = self._gen_gnrc_rltzn(rltzn_args)

            rltzns.append(rltzn)

            if not self._alg_ann_runn_auto_init_temp_search_flag:
                continue

            if self._sett_wts_obj_auto_set_flag:
                self._alg_auto_temp_init_obj_wts.append(
                    self._sett_wts_obj_wts.copy())

            pre_acpt_rates.append(rltzn[0])
            pre_init_temps.append(rltzn[1])

            if rltzn[0] >= self._sett_ann_auto_init_temp_acpt_bd_hi:
                break

            if rltzn[1] >= self._sett_ann_auto_init_temp_temp_bd_hi:
                break

        return rltzns


class PhaseAnnealingAlgTemperature:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _get_init_temp(
            self,
            auto_init_temp_atpt,
            pre_init_temps,
            pre_acpt_rates,
            init_temp):

        if self._alg_ann_runn_auto_init_temp_search_flag:

            assert isinstance(auto_init_temp_atpt, int), (
                'auto_init_temp_atpt not an integer!')

            assert (
                (auto_init_temp_atpt >= 0) and
                (auto_init_temp_atpt < self._sett_ann_auto_init_temp_atpts)), (
                    'Invalid _sett_ann_auto_init_temp_atpts!')

            assert len(pre_acpt_rates) == len(pre_init_temps), (
                'Unequal size of pre_acpt_rates and pre_init_temps!')

            if auto_init_temp_atpt:
                pre_init_temp = pre_init_temps[-1]
                pre_acpt_rate = pre_acpt_rates[-1]

                assert isinstance(pre_init_temp, float), (
                    'pre_init_temp not a float!')

                assert (
                    (pre_init_temp >=
                     self._sett_ann_auto_init_temp_temp_bd_lo) and
                    (pre_init_temp <=
                     self._sett_ann_auto_init_temp_temp_bd_hi)), (
                         'Invalid pre_init_temp!')

                assert isinstance(pre_acpt_rate, float), (
                    'pre_acpt_rate not a float!')

                assert 0 <= pre_acpt_rate <= 1, 'Invalid pre_acpt_rate!'

            if auto_init_temp_atpt == 0:
                init_temp = self._sett_ann_auto_init_temp_temp_bd_lo

            else:
                temp_lo_bd = self._sett_ann_auto_init_temp_temp_bd_lo
                temp_lo_bd *= (
                    self._sett_ann_auto_init_temp_ramp_rate **
                    (auto_init_temp_atpt - 1))

                temp_hi_bd = (
                    temp_lo_bd * self._sett_ann_auto_init_temp_ramp_rate)

                init_temp = temp_lo_bd + (
                    (temp_hi_bd - temp_lo_bd) * np.random.random())

                assert temp_lo_bd <= init_temp <= temp_hi_bd, (
                    'Invalid init_temp!')

                if init_temp > self._sett_ann_auto_init_temp_temp_bd_hi:
                    init_temp = self._sett_ann_auto_init_temp_temp_bd_hi

            assert (
                self._sett_ann_auto_init_temp_temp_bd_lo <=
                init_temp <=
                self._sett_ann_auto_init_temp_temp_bd_hi), (
                    'Invalid init_temp!')

        else:
            assert isinstance(init_temp, float), 'init_temp not a float!'
            assert 0 <= init_temp, 'Invalid init_temp!'

        return init_temp

    def _get_auto_init_temp(self):

        assert self._alg_verify_flag, 'Call verify first!'

        self._alg_ann_runn_auto_init_temp_search_flag = True

        if self._sett_wts_obj_auto_set_flag:
            self._alg_auto_temp_init_obj_wts = []

        acpt_rates_temps = np.atleast_2d(
            self._gen_gnrc_rltzns(
                ((0, self._sett_ann_auto_init_temp_atpts),)))

        best_acpt_rate_idx = np.argmin(
            (acpt_rates_temps[:, 0] -
             self._sett_ann_auto_init_temp_trgt_acpt_rate) ** 2)

        ann_init_temp = acpt_rates_temps[best_acpt_rate_idx, 1]

        assert (
            (self._sett_ann_auto_init_temp_temp_bd_lo <= ann_init_temp) &
            (self._sett_ann_auto_init_temp_temp_bd_hi >= ann_init_temp)), (
                'ann_init_temp out of bounds!')

        if self._sett_wts_obj_auto_set_flag:
            self._sett_wts_obj_wts = self._alg_auto_temp_init_obj_wts[
                best_acpt_rate_idx]

        self._alg_ann_runn_auto_init_temp_search_flag = False
        return ann_init_temp


class PhaseAnnealingAlgCDFIdxs:

    def _get_single_cdf_opt_idxs(self, raw_probs_dict):

        thresh_left_bd = 0.3
        thresh_rght_bd = 0.7
        buff_ratio = 0.02

        assert raw_probs_dict

        import matplotlib.pyplot as plt
        plt.ioff()

        cdf_opt_idxs = {}
        for key in raw_probs_dict:
            raw_probs_dict[key] = np.array(
                raw_probs_dict[key], order='f')

            raw_probs_dict[key].sort(axis=0)

            n_vals = raw_probs_dict[key].shape[1]

            cdf_opt_idxs[key] = np.zeros(n_vals, dtype=int)

            uf_probs = np.arange(1, n_vals + 1) / (n_vals + 1.0)

            thresh_lbd = round(n_vals * thresh_left_bd)
            thresh_ubd = round(n_vals * thresh_rght_bd)

            for i in range(n_vals):
                idx = np.searchsorted(
                    raw_probs_dict[key][:, i],
                    uf_probs[i])

                if thresh_lbd <= idx <= thresh_ubd:
                    continue

                cdf_opt_idxs[key][i] = 1

                buff_idx_lbd = max(0, int(i - (n_vals * buff_ratio)))
                buff_idx_ubd = min(n_vals, int(i + (n_vals * buff_ratio)))

                cdf_opt_idxs[key][buff_idx_lbd:buff_idx_ubd] = 1

            plt.plot(cdf_opt_idxs[key], alpha=0.5, label=key)

        plt.grid()
        plt.legend()

        plt.show()

        plt.close()

        return cdf_opt_idxs

    def _cmpt_cdf_opt_idxs(self):

        if self._vb:
            print_sl()

            print('Computing CDF opt. idxs...')

        self._alg_cdf_opt_n_sims = 1000

        self._alg_cdf_opt_idxs_flag = True

        if self._sett_obj_asymm_type_1_flag:
            self._alg_cdf_opt_asymms_1_sims = {}
            self._alg_cdf_opt_asymms_1_idxs = {}

        if self._sett_obj_asymm_type_2_flag:
            self._alg_cdf_opt_asymms_2_sims = {}
            self._alg_cdf_opt_asymms_2_idxs = {}

        i = 0
        while i < self._alg_cdf_opt_n_sims:
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(1.0, 1.0)

            self._update_sim(new_idxs, new_phss, new_coeffs)

            self._get_obj_ftn_val()

            i += 1

        if self._sett_obj_asymm_type_1_flag:
            self._alg_cdf_opt_asymms_1_idxs = self._get_single_cdf_opt_idxs(
                self._alg_cdf_opt_asymms_1_sims)

            self._alg_cdf_opt_asymms_1_sims = None

        if self._sett_obj_asymm_type_2_flag:
            self._alg_cdf_opt_asymms_2_idxs = self._get_single_cdf_opt_idxs(
                self._alg_cdf_opt_asymms_2_sims)

            self._alg_cdf_opt_asymms_2_sims = None

        if self._vb:
            print('Done computing CDF opt. idxs.')
            print_el()

        self._alg_cdf_opt_idxs_flag = False
        return


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
            self._sett_obj_pcorr_flag,
            self._sett_obj_asymm_type_1_ms_flag,
            self._sett_obj_asymm_type_2_ms_flag,
            self._sett_obj_ecop_dens_ms_flag,
            self._sett_obj_match_data_ft_flag,
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
         self._sett_obj_pcorr_flag,
         self._sett_obj_asymm_type_1_ms_flag,
         self._sett_obj_asymm_type_2_ms_flag,
         self._sett_obj_ecop_dens_ms_flag,
         self._sett_obj_match_data_ft_flag) = (
             [state] * self._sett_obj_n_flags)

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
         self._sett_obj_pcorr_flag,
         self._sett_obj_asymm_type_1_ms_flag,
         self._sett_obj_asymm_type_2_ms_flag,
         self._sett_obj_ecop_dens_ms_flag,
         self._sett_obj_match_data_ft_flag) = states

        assert len(states) == self._sett_obj_n_flags

        return

    def _trunc_interp_ftn(self, ftn_dict):

        for ftn in ftn_dict.values():
            ftn.x = ftn.x[1:-1]
            ftn.y = ftn.y[1:-1]
        return

    def _trunc_interp_ftns(self):

        '''
        Should only be called when all obj_ftn flags are True.
        '''

        assert self._sett_obj_use_obj_dist_flag

#         self._trunc_interp_ftn(self._ref_scorr_diffs_cdfs_dict)
# #         self._trunc_interp_ftn(self._ref_asymm_1_diffs_cdfs_dict)
# #         self._trunc_interp_ftn(self._ref_asymm_2_diffs_cdfs_dict)
#         self._trunc_interp_ftn(self._ref_ecop_dens_diffs_cdfs_dict)
#         self._trunc_interp_ftn(self._ref_ecop_etpy_diffs_cdfs_dict)
# #         self._trunc_interp_ftn(self._ref_nth_ord_diffs_cdfs_dict)
#         self._trunc_interp_ftn(self._ref_pcorr_diffs_cdfs_dict)
#
#         if self._data_ref_n_labels > 1:
#             self._trunc_interp_ftn(self._ref_mult_asymm_1_diffs_cdfs_dict)
#             self._trunc_interp_ftn(self._ref_mult_asymm_2_diffs_cdfs_dict)

        return

    def _update_ref_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        self._prep_vld_flag = True

        self._gen_ref_aux_data()

        if trunc_interp_ftns_flag:
            self._trunc_interp_ftns()

        self._prep_vld_flag = False

        self._set_all_flags_to_mult_states(old_flags)
        return

    def _update_sim_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        self._prep_vld_flag = True

        # Calling self._gen_sim_aux_data creates a problem by randomizing
        # everything again. Hence, the call to self._update_obj_vars.

        self._sim_ft = self._sim_ft_best

        self._sim_phs_spec = np.angle(self._sim_ft)
        self._sim_mag_spec = np.abs(self._sim_ft)

        self._update_sim_no_prms()

        self._prep_vld_flag = False

        self._set_all_flags_to_mult_states(old_flags)
        return


class PhaseAnnealingAlgorithm(
        PAP,
        PhaseAnnealingAlgObjective,
        PhaseAnnealingAlgIO,
        PhaseAnnealingAlgRealization,
        PhaseAnnealingAlgTemperature,
        PhaseAnnealingAlgCDFIdxs,
        PhaseAnnealingAlgMisc):

    '''The main phase annealing class'''

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_ann_runn_auto_init_temp_search_flag = False

        self._lock = None

        self._alg_rltzns_gen_flag = False

        self._alg_force_acpt_flag = False

        self._alg_auto_temp_init_obj_wts = None

        self._alg_done_opt_flag = False
        self._alg_pnl_flag = False

        self._alg_asymm_1_dist_sclr = 1.0
        self._alg_asymm_1_dist_sclr_slp = -0.001

        self._alg_asymm_2_dist_sclr = 1.0
        self._alg_asymm_2_dist_sclr_slp = -0.001

        # CDF opt idxs.
        self._alg_cdf_opt_n_sims = None

        self._alg_cdf_opt_asymms_1_sims = None
        self._alg_cdf_opt_asymms_1_idxs = None

        self._alg_cdf_opt_asymms_2_idxs = None
        self._alg_cdf_opt_asymms_2_sims = None

        self._alg_cdf_opt_idxs_flag = False
        self._alg_verify_flag = False
        return

    def simulate(self):

        '''Start the phase annealing algorithm'''

        beg_sim_tm = default_timer()

        self._init_output()

        self._write_non_sim_data_to_h5()

        if self._vb:
            print_sl()

            print('Starting simulations...')

            print_el()

            print('\n')

        if self._sett_cdf_opt_idxs_flag:
            self._cmpt_cdf_opt_idxs()

        if self._sett_misc_n_cpus > 1:

            mp_idxs = ret_mp_idxs(
                self._sett_misc_n_rltzns, self._sett_misc_n_cpus)

            rltzns_gen = (
                (
                (mp_idxs[i], mp_idxs[i + 1]),
                )
                for i in range(mp_idxs.size - 1))

            self._lock = Manager().Lock()

            mp_pool = ProcessPool(self._sett_misc_n_cpus)
            mp_pool.restart(True)

            list(mp_pool.uimap(self._sim_grp, rltzns_gen))

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

        else:
            self._lock = Lock()

            self._sim_grp(((0, self._sett_misc_n_rltzns),))

            self._lock = None

        end_sim_tm = default_timer()

        if self._vb:
            print_sl()

            print(
                f'Total simulation time was: '
                f'{end_sim_tm - beg_sim_tm:0.3f} seconds!')

            print_el()

        self._alg_rltzns_gen_flag = True
        return

    def verify(self):

        PAP._PhaseAnnealingPrepare__verify(self)
        assert self._prep_verify_flag, 'Prepare in an unverified state!'

        if self._vb:
            print_sl()

            print(
                'Phase annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    def _init_output(self):

        if self._vb:
            print_sl()

            print('Initializing outputs file...')

        if not self._sett_misc_outs_dir.exists():
            self._sett_misc_outs_dir.mkdir(exist_ok=True)

        assert self._sett_misc_outs_dir.exists(), (
            'Could not create outputs_dir!')

        h5_path = self._sett_misc_outs_dir / self._save_h5_name

        # Create new / overwrite old.
        # It's simpler to do it here.
        h5_hdl = h5py.File(h5_path, mode='w', driver=None)
        h5_hdl.close()

        if self._vb:
            print('Initialized the outputs file.')

            print_el()
        return

    def _update_phs_ann_cls_vars(self):

        # ref cls update
#         self._ref_phs_ann_class_vars[0] = (
#             self._ref_phs_ann_class_vars[1])
#
#         self._ref_phs_ann_class_vars[1] += (
#             self._sett_ann_phs_ann_class_width)
#
#         self._ref_phs_ann_class_vars[1] = min(
#             self._ref_phs_ann_class_vars[1],
#             self._ref_mag_spec.shape[0])
#
#         self._ref_phs_ann_class_vars[3] += 1

        self._ref_phs_ann_class_vars[1] = (
            self._ref_phs_ann_class_vars[0])

        self._ref_phs_ann_class_vars[0] -= (
            self._sett_ann_phs_ann_class_width)

        self._ref_phs_ann_class_vars[0] = max(
            self._ref_phs_ann_class_vars[0],
            1)

        self._ref_phs_ann_class_vars[3] += 1

        # sim cls update
#         self._sim_phs_ann_class_vars[0] = (
#             self._sim_phs_ann_class_vars[1])
#
#         self._sim_phs_ann_class_vars[1] += (
#             self._sett_ann_phs_ann_class_width *
#             self._sett_extnd_len_rel_shp[0])
#
#         self._sim_phs_ann_class_vars[1] = min(
#             self._sim_phs_ann_class_vars[1],
#             self._sim_mag_spec.shape[0])
#
#         self._sim_phs_ann_class_vars[3] += 1

        self._sim_phs_ann_class_vars[1] = (
            self._sim_phs_ann_class_vars[0])

        self._sim_phs_ann_class_vars[0] -= (
            self._sett_ann_phs_ann_class_width *
            self._sett_extnd_len_rel_shp[0])

        self._sim_phs_ann_class_vars[0] = max(
            self._sim_phs_ann_class_vars[0],
            1)

        self._sim_phs_ann_class_vars[3] += 1

        return

    def _sim_grp(self, args):

        ((beg_rltzn_iter, end_rltzn_iter),
        ) = args

        beg_thread_time = default_timer()

        for rltzn_iter in range(beg_rltzn_iter, end_rltzn_iter):
            if self._vb:
                with self._lock:
                    print(f'Started realization {rltzn_iter}...')

            beg_tot_rltzn_tm = default_timer()

            self._set_phs_ann_cls_vars_ref()
            self._set_phs_ann_cls_vars_sim()

            while (
                self._sim_phs_ann_class_vars[3] <
                self._sim_phs_ann_class_vars[2]):

                self._reset_timers()

                beg_cls_tm = default_timer()

                self._gen_ref_aux_data()

                if self._sett_auto_temp_set_flag:
                    beg_it_tm = default_timer()

                    init_temp = self._get_auto_init_temp()

                    end_it_tm = default_timer()

                    if self._vb:
                        with self._lock:
                            print(
                                f'Initial temperature ({init_temp:06.4e}) '
                                f'computation took '
                                f'{end_it_tm - beg_it_tm:0.3f} '
                                f'seconds for realization {rltzn_iter} and '
                                f'class {self._sim_phs_ann_class_vars[3]}.')

                else:
                    init_temp = self._sett_ann_init_temp

                beg_rltzn_tm = default_timer()

                stopp_criteria = self._gen_gnrc_rltzn(
                    (rltzn_iter, None, None, init_temp))

                end_rltzn_tm = default_timer()

                end_cls_tm = default_timer()

                if self._vb:
                    with self._lock:

                        print('\n')

                        print(
                            f'Realization {rltzn_iter} for class '
                            f'{self._sim_phs_ann_class_vars[3]} '
                            f'computation took '
                            f'{end_rltzn_tm - beg_rltzn_tm:0.3f} '
                            f'seconds with stopp_criteria: '
                            f'{stopp_criteria}.')

                        print(
                            f'Realization {rltzn_iter} for class '
                            f'{self._sim_phs_ann_class_vars[3]} '
                            f'took a total of '
                            f'{end_cls_tm - beg_cls_tm:0.3f} seconds.')

                self._update_phs_ann_cls_vars()

            end_tot_rltzn_tm = default_timer()

            self._reset_timers()

            assert np.all(self._sim_phs_mod_flags >= 1), (
                'Some phases were not modified!')

            if self._vb:
                with self._lock:
                    print(
                        f'Realization {rltzn_iter} took '
                        f'{end_tot_rltzn_tm - beg_tot_rltzn_tm:0.3f} '
                        f'seconds for all classes.\n')

        end_thread_time = default_timer()

        if self._vb:
            with self._lock:
                print(
                    f'Total thread time for realizations between '
                    f'{beg_rltzn_iter} and {end_rltzn_iter - 1} was '
                    f'{end_thread_time - beg_thread_time:0.3f} '
                    f'seconds.')
        return

    __verify = verify

