'''
Created on Dec 27, 2019

@author: Faizan
'''
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


class PhaseAnnealingAlgObjective:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.

    obj_vals are normalized in a way to have them comparable amongst all
    sorts of objective functions' or other settings' combinations.
    '''

    def _get_obj_scorr_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:

                    sim_diffs = self._sim_scorr_diffs[(label, lag)]

                    ftn = self._ref_scorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diffs.sum()  # / ftn.sclr

        else:
            obj_val = (
                ((self._ref_scorrs - self._sim_scorrs) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_asymms_1_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:
                    sim_diffs = self._sim_asymm_1_diffs[(label, lag)]

                    ftn = self._ref_asymm_1_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diffs.sum()  # / ftn.sclr

        else:
            obj_val = (
                ((self._ref_asymms_1 - self._sim_asymms_1) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_asymms_2_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:

                    sim_diffs = self._sim_asymm_2_diffs[(label, lag)]

                    ftn = self._ref_asymm_2_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diffs.sum()  # / ftn.sclr

        else:
            obj_val = (
                ((self._ref_asymms_2 - self._sim_asymms_2) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_ecop_dens_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:

                    sim_diffs = self._sim_ecops_dens_diffs[(label, lag)]

                    ftn = self._ref_ecop_dens_diffs_cdfs_dict[(label, lag)]

                    sim_probs = ftn(sim_diffs)

                    ref_probs = ftn.y

                    sq_diff = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diff.sum()  # / ftn.sclr

        else:
            obj_val = ((
                (self._ref_ecop_dens_arrs -
                 self._sim_ecop_dens_arrs) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_ecop_etpy_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:

                    sim_diffs = self._sim_ecops_etpy_diffs[(label, lag)]

                    ftn = self._ref_ecop_etpy_diffs_cdfs_dict[(label, lag)]

                    sim_probs = ftn(sim_diffs)

                    ref_probs = ftn.y

                    sq_diff = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diff.sum() / ftn.sclr

        else:
            obj_val = ((
                (self._ref_ecop_etpy_arrs -
                 self._sim_ecop_etpy_arrs) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_nth_ord_diffs_val(self):

        obj_val = 0.0
        for label in self._data_ref_labels:
            for nth_ord in self._sett_obj_nth_ords:

                sim_diffs = self._sim_nth_ord_diffs[(label, nth_ord)]

                ftn = self._ref_nth_ords_cdfs_dict[(label, nth_ord)]

                ref_probs = ftn.y

                sim_probs = ftn(sim_diffs)

                sq_diffs = ((ref_probs - sim_probs) * ftn.wts) ** 2

                obj_val += sq_diffs.sum()  # / ftn.sclr

        return obj_val

    def _get_obj_cos_sin_dist_val(self):

        obj_val = 0.0

        for i, label in enumerate(self._data_ref_labels):
            cos_ftn = self._ref_cos_sin_dists_dict[(label, 'cos')]
            ref_probs_cos = cos_ftn.y
            sim_probs_cos = np.sort(cos_ftn(self._sim_ft.real[:, i]))
            cos_sq_diffs = ((ref_probs_cos - sim_probs_cos) * cos_ftn.wts) ** 2
            obj_val += cos_sq_diffs.sum()  # / cos_ftn.sclr

            sin_ftn = self._ref_cos_sin_dists_dict[(label, 'sin')]
            ref_probs_sin = sin_ftn.y
            sim_probs_sin = np.sort(sin_ftn(self._sim_ft.imag[:, i]))
            sin_sq_diffs = ((ref_probs_sin - sim_probs_sin) * sin_ftn.wts) ** 2
            obj_val += sin_sq_diffs.sum()  # / sin_ftn.sclr

        return obj_val

    def _get_obj_pcorr_val(self):

        if self._sett_obj_use_obj_dist_flag:
            obj_val = 0.0
            for label in self._data_ref_labels:
                for lag in self._sett_obj_lag_steps:

                    sim_diffs = self._sim_pcorr_diffs[(label, lag)]

                    ftn = self._ref_pcorr_diffs_cdfs_dict[(label, lag)]

                    ref_probs = ftn.y

                    sim_probs = ftn(sim_diffs)

                    sq_diffs = ((ref_probs - sim_probs) * ftn.wts) ** 2

                    obj_val += sq_diffs.sum()  # / ftn.sclr

        else:
            obj_val = (
                ((self._ref_pcorrs - self._sim_pcorrs) ** 2).sum() /
                (self._data_ref_n_labels * self._sett_obj_lag_steps.size))

        return obj_val

    def _get_obj_pve_dgnl_val(self):

        return

    def _get_obj_ftn_val(self):

        obj_val = 0.0

        if self._sett_obj_scorr_flag:
            obj_val += self._get_obj_scorr_val()

        if self._sett_obj_asymm_type_1_flag:
            obj_val += self._get_obj_asymms_1_val()

        if self._sett_obj_asymm_type_2_flag:
            obj_val += self._get_obj_asymms_2_val()

        if self._sett_obj_ecop_dens_flag:
            obj_val += self._get_obj_ecop_dens_val()

        if self._sett_obj_ecop_etpy_flag:
            obj_val += self._get_obj_ecop_etpy_val()

        if self._sett_obj_nth_ord_diffs_flag:
            obj_val += self._get_obj_nth_ord_diffs_val()

        if self._sett_obj_cos_sin_dist_flag:
            obj_val += self._get_obj_cos_sin_dist_val()

        if self._sett_obj_pcorr_flag:
            obj_val += self._get_obj_pcorr_val()

        if self._sett_obj_pve_dgnl_flag:
            obj_val += self._get_obj_pve_dgnl_val()

        assert np.isfinite(obj_val), 'Invalid obj_val!'

        return obj_val


class PhaseAnnealingAlgIO:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _write_cls_rltzn(self, rltzn_iter, ret):

        with self._lock:
            # _update_ref_at_end called inside _write_cls_rltzn

            h5_path = self._sett_misc_outs_dir / self._save_h5_name

            with h5py.File(h5_path, mode='a', driver=None) as h5_hdl:
                self._write_ref_cls_rltzn(h5_hdl)
                self._write_sim_cls_rltzn(h5_hdl, rltzn_iter, ret)
        return

    def _write_ref_cls_rltzn(self, h5_hdl):

        # should be called by _write_cls_rltzn with a lock

        cls_pad_zeros = len(str(self._ref_phs_ann_class_vars[2]))

        ref_cls_grp_lab = (
            f'data_ref_rltzn/'
            f'{self._ref_phs_ann_class_vars[3]:0{cls_pad_zeros}d}')

        if ref_cls_grp_lab in h5_hdl:
            return

        self._update_ref_at_end()

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

                  all([isinstance(key[lg_idx], np.int64) for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    ref_cls_grp[
                        data_lab + f'_{key[ll_idx]}_{key[lg_idx]:03d}_x'] = data_val[key].x

                    ref_cls_grp[
                        data_lab + f'_{key[ll_idx]}_{key[lg_idx]:03d}_y'] = data_val[key].y

            elif (isinstance(data_val, dict) and

                  all([key[lg_idx] in ('cos', 'sin') for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    ref_cls_grp[data_lab + f'_{key[ll_idx]}_{key[lg_idx]}_x'] = data_val[key].x
                    ref_cls_grp[data_lab + f'_{key[ll_idx]}_{key[lg_idx]}_y'] = data_val[key].y

            elif (isinstance(data_val, dict) and

                  all([isinstance(key[lg_idx], np.int64) for key in data_val]) and

                  all([isinstance(val, np.ndarray)
                       for val in data_val.values()])):

                for key in data_val:
                    ref_cls_grp[data_lab + f'_{key[ll_idx]}_{key[lg_idx]:03d}'] = data_val[key]

            elif (isinstance(data_val, dict) and

                  all([all([col in self._data_ref_labels for col in key]) for key in data_val]) and

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
            (not np.isclose(temp, 0.0)),
#             (temp > 1e-15),
            (not np.isclose(phs_red_rate, 0.0)),
#             (phs_red_rate > 1e-15),
            (acpt_rate > self._sett_ann_stop_acpt_rate),
            )

        return stopp_criteria

    def _get_next_idxs(self):

        # _sim_mag_spec_cdf makes it difficult without a while-loop.

        # TODO: Generated indices should be close to each other to simulate
        # a discharge rise and recession correctly. Don't know how it works
        # for multiple events.

        # Inclusive
        min_idxs_to_gen = min([
            self._sett_mult_phs_n_beg_phss,
            self._sim_phs_ann_class_vars[1] - self._sim_phs_ann_class_vars[0]])

        # Inclusive
        max_idxs_to_gen = min([
            self._sett_mult_phs_n_end_phss,
            self._sim_phs_ann_class_vars[1] - self._sim_phs_ann_class_vars[0]])

        idxs_to_gen = np.random.randint(min_idxs_to_gen, max_idxs_to_gen + 1)

        max_ctr = 100 * self._sim_shape[0] * self._data_ref_n_labels

        new_idxs = []
        while len(new_idxs) < idxs_to_gen:

            idx_ctr = 0
            while True:

                if self._sett_ann_mag_spec_cdf_idxs_flag:
                    index = int(self._sim_mag_spec_cdf(np.random.random()))

                else:
                    index = int(np.random.random() * (self._sim_shape[0] - 1))

                assert 0 <= index <= self._sim_shape[0], (
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

    def _get_next_iter_vars(self, phs_red_rate):

        new_idxs = self._get_next_idxs()

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

    def _update_sim(self, idxs, phss, coeffs):

        self._sim_phs_spec[idxs] = phss

        if coeffs is not None:
            self._sim_ft[idxs] = coeffs
            self._sim_mag_spec[idxs] = np.abs(self._sim_ft[idxs])

        else:
            self._sim_ft.real[idxs] = np.cos(phss) * self._sim_mag_spec[idxs]
            self._sim_ft.imag[idxs] = np.sin(phss) * self._sim_mag_spec[idxs]

        data = np.fft.irfft(self._sim_ft, axis=0)

        probs, norms = self._get_probs_norms(data, True)

        self._sim_data = np.empty_like(
            self._data_ref_rltzn_srtd, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            self._sim_data[:, i] = self._data_ref_rltzn_srtd[
                np.argsort(np.argsort(probs[:, i])), i]

        self._sim_probs = probs
        self._sim_nrm = norms

        self._update_obj_vars('sim')
        return

    def _gen_gnrc_rltzn(self, args):

        (rltzn_iter,
         pre_init_temps,
         pre_acpt_rates,
         init_temp) = args

        assert isinstance(rltzn_iter, int), 'rltzn_iter not integer!'

        if self._alg_ann_runn_auto_init_temp_search_flag:
            assert 0 <= rltzn_iter < self._sett_ann_auto_init_temp_atpts, (
                    'Invalid rltzn_iter!')

        else:
            assert 0 <= rltzn_iter < self._sett_misc_n_rltzns, (
                    'Invalid rltzn_iter!')

        if self._data_ref_rltzn.ndim != 2:
            raise NotImplementedError('Implemention for 2D only!')

        # randomize all phases before starting
        self._gen_sim_aux_data()

        # initialize sim anneal variables
        iter_ctr = 0
        phs_red_rate = 1.0

        temp = self._get_init_temp(
            rltzn_iter, pre_init_temps, pre_acpt_rates, init_temp)

        old_idxs = self._get_next_idxs()
        new_idxs = old_idxs

        old_obj_val = self._get_obj_ftn_val()

        if self._alg_ann_runn_auto_init_temp_search_flag:
            stopp_criteria = (
                (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                )

        else:
            iters_wo_acpt = 0
            tol = np.inf
            acpt_rate = 1.0

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

        # initialize diagnostic variables
        acpts_rjts_all = []

        if not self._alg_ann_runn_auto_init_temp_search_flag:
            tols = []

            obj_vals_all = []

            obj_val_min = np.inf
            obj_vals_min = []

            idxs_all = []
            idxs_acpt = []

            phss_all = []
            phs_red_rates = [[iter_ctr, phs_red_rate]]

            temps = [[iter_ctr, temp]]

            acpt_rates_dfrntl = [[iter_ctr, acpt_rate]]

        else:
            pass

        while all(stopp_criteria):

            #==============================================================
            # Simulated annealing start
            #==============================================================

            (old_phss,
             new_phss,
             old_coeffs,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate)

            self._update_sim(new_idxs, new_phss, new_coeffs)

            new_obj_val = self._get_obj_ftn_val()

#             print(new_obj_val, old_obj_val, old_phs, new_phs, old_index, new_index)

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

            if accept_flag:
                old_idxs = new_idxs

                old_obj_val = new_obj_val

            else:
                self._update_sim(new_idxs, old_phss, old_coeffs)

            iter_ctr += 1

            #==============================================================
            # Simulated annealing end
            #==============================================================

            acpts_rjts_all.append(accept_flag)

            if self._alg_ann_runn_auto_init_temp_search_flag:
                stopp_criteria = (
                    (iter_ctr <= self._sett_ann_auto_init_temp_niters),
                    )

            else:
                tols_dfrntl.append(abs(old_new_diff))

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    obj_val_min = min(obj_val_min, old_obj_val)

                obj_vals_min.append(obj_val_min)
                obj_vals_all.append(new_obj_val)

                acpts_rjts_dfrntl.append(accept_flag)

                phss_all.append(new_phss)
                idxs_all.append(new_idxs)

                if iter_ctr >= tols_dfrntl.maxlen:
                    tol = sum(tols_dfrntl) / float(tols_dfrntl.maxlen)

                    assert np.isfinite(tol), 'Invalid tol!'

                    tols.append(tol)

                if accept_flag:
                    idxs_acpt.append((iter_ctr - 1, new_idxs))

                    iters_wo_acpt = 0

                else:
                    iters_wo_acpt += 1

                if iter_ctr >= acpts_rjts_dfrntl.maxlen:
                    acpt_rates_dfrntl.append([iter_ctr - 1, acpt_rate])

                    acpt_rate = (
                        sum(acpts_rjts_dfrntl) /
                        float(acpts_rjts_dfrntl.maxlen))

                    acpt_rates_dfrntl.append([iter_ctr, acpt_rate])

                if not (iter_ctr % self._sett_ann_upt_evry_iter):

                    temps.append([iter_ctr - 1, temp])

                    temp *= self._sett_ann_temp_red_rate

                    assert temp >= 0.0, 'Invalid temp!'

                    temps.append([iter_ctr, temp])

                    phs_red_rates.append([iter_ctr - 1, phs_red_rate])

                    if self._sett_ann_phs_red_rate_type == 0:
                        pass

                    elif self._sett_ann_phs_red_rate_type == 1:
                        phs_red_rate = (
                            (self._sett_ann_max_iters - iter_ctr) /
                            self._sett_ann_max_iters)

                    elif self._sett_ann_phs_red_rate_type == 2:
                        phs_red_rate *= self._sett_ann_phs_red_rate

                    elif self._sett_ann_phs_red_rate_type == 3:

                        # An unstable mean of acpts_rjts_dfrntl is a
                        # problem. So, it has to be long enough.
                        phs_red_rate = acpt_rate

                    else:
                        raise NotImplemented(
                            'Unknown _sett_ann_phs_red_rate_type:',
                            self._sett_ann_phs_red_rate_type)

                    assert phs_red_rate >= 0.0, 'Invalid phs_red_rate!'

                    phs_red_rates.append([iter_ctr, phs_red_rate])

                stopp_criteria = self._get_stopp_criteria(
                    (iter_ctr,
                     iters_wo_acpt,
                     tol,
                     temp,
                     phs_red_rate,
                     acpt_rate))

        if self._alg_ann_runn_auto_init_temp_search_flag:
            ret = (sum(acpts_rjts_all) / len(acpts_rjts_all), temp)

        else:
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
                self._sim_nrm,
                self._sim_scorrs,
                self._sim_asymms_1,
                self._sim_asymms_2,
                self._sim_ecop_dens_arrs,
                self._sim_ecop_etpy_arrs,
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
#                 np.array(phss_all, dtype=np.float64),
                np.array(temps, dtype=np.float64),
                np.array(phs_red_rates, dtype=np.float64),
#                 np.array(idxs_all, dtype=np.uint64),
#                 np.array(idxs_acpt, dtype=np.uint64),
                np.array(acpt_rates_dfrntl, dtype=np.float64),
                ref_sim_ft_corr,
                sim_sim_ft_corr,
#                 self._sim_phs_cross_corr_mat,  # not of any use
                self._sim_phs_ann_class_vars,
                self._sim_data,
                self._sim_pcorrs,
                self._sim_phs_mod_flags,
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
                [val for val in self._sim_ecops_dens_diffs.values()])

            out_data.extend(
                [val for val in self._sim_ecops_etpy_diffs.values()])

            out_data.extend(
                [val for val in self._sim_pcorr_diffs.values()])

            if self._ref_mult_asymm_1_diffs_cdfs_dict is not None:
                out_data.extend(
                    [val for val in self._sim_mult_asymms_1_diffs.values()])

            if self._ref_mult_asymm_2_diffs_cdfs_dict is not None:
                out_data.extend(
                    [val for val in self._sim_mult_asymms_2_diffs.values()])

            # _update_ref_at_end called inside _write_cls_rltzn if needed.
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

        self._alg_ann_runn_auto_init_temp_search_flag = False
        return ann_init_temp


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
            self._sett_obj_pve_dgnl_flag)

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
         self._sett_obj_pve_dgnl_flag) = [state] * self._sett_obj_n_flags

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
         self._sett_obj_pcorr_flag) = states

        assert len(states) == self._sett_obj_n_flags

        return

    def _update_ref_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        self._gen_ref_aux_data()

#         self._ref_phs_cross_corr_mat = self._get_phs_cross_corr_mat(
#             self._ref_phs_spec)

        self._set_all_flags_to_mult_states(old_flags)
        return

    def _update_sim_at_end(self):

        old_flags = self._get_all_flags()

        self._set_all_flags_to_one_state(True)

        # Calling self._gen_sim_aux_data creates a problem by randomizing
        # everything again. Hence, the call to self._update_obj_vars.

        self._update_obj_vars('sim')

#         self._sim_phs_cross_corr_mat = self._get_phs_cross_corr_mat(
#             self._sim_phs_spec)

        self._set_all_flags_to_mult_states(old_flags)
        return


class PhaseAnnealingAlgorithm(
        PAP,
        PhaseAnnealingAlgObjective,
        PhaseAnnealingAlgIO,
        PhaseAnnealingAlgRealization,
        PhaseAnnealingAlgTemperature,
        PhaseAnnealingAlgMisc):

    '''The main phase annealing class'''

    def __init__(self, verbose=True):

        PAP.__init__(self, verbose)

        self._alg_ann_runn_auto_init_temp_search_flag = False

        self._lock = None

        self._alg_rltzns_gen_flag = False

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

        # create new / overwrite old.
        # It's simpler to do it here.
        h5_hdl = h5py.File(h5_path, mode='w', driver=None)
        h5_hdl.close()

        if self._vb:
            print('Initialized the outputs file.')

            print_el()
        return

    def _update_phs_ann_cls_vars(self):

        # ref cls update
        self._ref_phs_ann_class_vars[0] = (
            self._ref_phs_ann_class_vars[1])

        self._ref_phs_ann_class_vars[1] += (
            self._sett_ann_phs_ann_class_width)

        self._ref_phs_ann_class_vars[1] = min(
            self._ref_phs_ann_class_vars[1],
            self._ref_mag_spec.shape[0])

        self._ref_phs_ann_class_vars[3] += 1

        # sim cls update
        self._sim_phs_ann_class_vars[0] = (
            self._sim_phs_ann_class_vars[1])

        self._sim_phs_ann_class_vars[1] += (
            self._sett_ann_phs_ann_class_width *
            self._sett_extnd_len_rel_shp[0])

        self._sim_phs_ann_class_vars[1] = min(
            self._sim_phs_ann_class_vars[1],
            self._sim_mag_spec.shape[0])

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

                beg_cls_tm = default_timer()

                self._gen_ref_aux_data()

                if self._sett_auto_temp_set_flag:
                    beg_it_tm = default_timer()

                    init_temp = self._get_auto_init_temp()

                    end_it_tm = default_timer()

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

                        if self._sett_auto_temp_set_flag:
                            print(
                                f'Initial temperature computation took '
                                f'{end_it_tm - beg_it_tm:0.3f} '
                                f'seconds for realization {rltzn_iter} and '
                                f'class {self._sim_phs_ann_class_vars[3]}.')

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

