'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from math import ceil
from timeit import default_timer
from itertools import product

import h5py
import numpy as np
from KDEpy import FFTKDE
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from matplotlib.colors import Normalize

from ...misc import roll_real_2arrs
from .setts import get_mpl_prms, set_mpl_prms

plt.ioff()


class PhaseAnnealingPlotSingleSite:

    '''
    Supporting class of Plot. Doesn't have __init__ of it own.

    Single-site plots.
    '''

    def _plot_cmpr_etpy_ft(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, lag_steps)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, lag_step) in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_etpy_ft = ref_grp[
                f'_ref_etpy_ft_dict_{data_label}_{lag_step:03d}'][:]

            ref_periods = (ref_etpy_ft.size * 2) / (
                np.arange(1, ref_etpy_ft.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_probs_ft = sim_grp_main[
                    f'{rltzn_lab}/etpy_ft_{data_label}_{lag_step:03d}'][:]

                if sim_periods is None:
                    sim_periods = (sim_probs_ft.size * 2) / (
                        np.arange(1, sim_probs_ft.size + 1))

                plt.semilogx(
                    sim_periods,
                    sim_probs_ft,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.semilogx(
                ref_periods,
                ref_etpy_ft,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            plt.grid()

            if lag_step in lag_steps_opt:
                suff = 'opt'

            else:
                suff = 'vld'

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative etpy FT correlation')

            plt.xlabel(f'Period (steps), (lag step(s) = {lag_step}_{suff})')

            plt.xlim(plt.xlim()[::-1])

            out_name = f'ss__etpy_ft_{data_label}_{lag_step:03d}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site etpy FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_diffs_ft_nth_ords(self, var_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ss__{var_label}_diffs_ft_cumsum'

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords_vld']
        nth_ords_opt = h5_hdl['settings/_sett_obj_nth_ords']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, nth_ords)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, nth_ord) in loop_prod:

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_ft_'
                f'dict_{data_label}_{nth_ord:03d}'][:]

            ref_periods = ((ref_vals.size - 1) * 2) / (
                np.arange(1, ref_vals.size))

            ref_periods = np.concatenate(([ref_periods[0] * 2], ref_periods))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_vals,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/{var_label}_'
                    f'diffs_ft_{data_label}_{nth_ord:03d}'][:]

                if sim_periods is None:
                    sim_periods = ((sim_vals.size - 1) * 2) / (
                        np.arange(1, sim_vals.size))

                    sim_periods = np.concatenate(
                        ([sim_periods[0] * 2], sim_periods))

                plt.semilogx(
                    sim_periods,
                    sim_vals,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.xlim(plt.xlim()[::-1])

            if nth_ord in nth_ords_opt:
                suff = 'opt'

            else:
                suff = 'vld'

            plt.ylabel('Cummulative diffs FT correlation')
            plt.xlabel(f'Period (steps), (lag step(s) = {nth_ord}_{suff})')

            out_name = f'{out_name_pref}_{data_label}_{nth_ord:03d}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site {var_label} diffs FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_diffs_ft_lags(self, var_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ss__{var_label}_diffs_ft_cumsum'

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, lag_steps)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, lag_step) in loop_prod:

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_ft_'
                f'dict_{data_label}_{lag_step:03d}'][:]

            ref_periods = ((ref_vals.size - 1) * 2) / (
                np.arange(1, ref_vals.size))

            ref_periods = np.concatenate(([ref_periods[0] * 2], ref_periods))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_vals,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/{var_label}_'
                    f'diffs_ft_{data_label}_{lag_step:03d}'][:]

                if sim_periods is None:
                    sim_periods = ((sim_vals.size - 1) * 2) / (
                        np.arange(1, sim_vals.size))

                    sim_periods = np.concatenate(
                        ([sim_periods[0] * 2], sim_periods))

                plt.semilogx(
                    sim_periods,
                    sim_vals,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.xlim(plt.xlim()[::-1])

            if lag_step in lag_steps_opt:
                suff = 'opt'

            else:
                suff = 'vld'

            plt.ylabel('Cummulative diffs FT correlation')
            plt.xlabel(f'Period (steps), (lag step(s) = {lag_step}_{suff})')

            out_name = f'{out_name_pref}_{data_label}_{lag_step:03d}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site {var_label} diffs FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_probs_ft(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_probs_ft = ref_grp['_ref_probs_ft'][:, data_lab_idx]

            ref_periods = (ref_probs_ft.size * 2) / (
                np.arange(1, ref_probs_ft.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_probs_ft,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_probs_ft = sim_grp_main[
                    f'{rltzn_lab}/probs_ft'][:, data_lab_idx]

                if sim_periods is None:
                    sim_periods = (sim_probs_ft.size * 2) / (
                        np.arange(1, sim_probs_ft.size + 1))

                plt.semilogx(
                    sim_periods,
                    sim_probs_ft,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative probs FT correlation')

            plt.xlabel(f'Period (steps)')

            plt.xlim(plt.xlim()[::-1])

            out_name = f'ss__probs_ft_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site probs FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_data_ft(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_data_ft = ref_grp['_ref_data_ft'][:, data_lab_idx]

            ref_periods = (ref_data_ft.size * 2) / (
                np.arange(1, ref_data_ft.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_data_ft,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_data_ft = sim_grp_main[
                    f'{rltzn_lab}/data_ft'][:, data_lab_idx]

                if sim_periods is None:
                    sim_periods = (sim_data_ft.size * 2) / (
                        np.arange(1, sim_data_ft.size + 1))

                plt.semilogx(
                    sim_periods,
                    sim_data_ft,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative data FT correlation')

            plt.xlabel(f'Period (steps)')

            plt.xlim(plt.xlim()[::-1])

            out_name = f'ss__data_ft_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site data FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _get_dens_ftn(self, probs, vals):

        '''
        NOTE: PDFs turned to be too peaked and therefore of no use.
        '''

        assert probs.size == vals.size

#         finer_vals = np.linspace(vals[+0], vals[-1], vals.size * 2)
#
#         interp_ftn = interp1d(
#             vals,
#             probs,
#             kind='slinear')
#
#         finer_probs = interp_ftn(
#             (finer_vals + ((finer_vals[1] - finer_vals[0]) * 0.5))[:-1])
#
#         finer_vals = finer_vals[1:-1]
#
#         finer_probs_diff = finer_probs[1:] - finer_probs[:-1]
#
#         dens = finer_probs_diff / (finer_vals[1] - finer_vals[0])
#         dens[dens < 0] = 0
#         dens /= dens.sum()
#
#         assert dens.size == finer_vals.size
#
#         return dens, finer_vals

        new_vals, dens = FFTKDE(bw="silverman").fit(vals).evaluate(vals.size)

        return dens, new_vals

    def _plot_gnrc_cdfs_cmpr(self, var_label, x_ax_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ss__{var_label}_diff_cdfs'

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld'][:]
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps'][:]
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, lag_steps)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, lag_step) in loop_prod:

            ref_probs = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{data_label}_{lag_step:03d}_y'][:]

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{data_label}_{lag_step:03d}_x'][:]

            if self._dens_dist_flag:
                ref_probs_plt, ref_vals_plt = self._get_dens_ftn(
                    ref_probs, ref_vals)

            else:
                ref_probs_plt, ref_vals_plt = ref_probs, ref_vals

            plt.figure()

            plt.plot(
                ref_vals_plt,
                ref_probs_plt,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/{var_label}_'
                    f'diffs_{data_label}_{lag_step:03d}'][:]

                sim_probs = rankdata(sim_vals) / (sim_vals.size + 1)

                if self._dens_dist_flag:
                    sim_probs_plt, sim_vals_plt = self._get_dens_ftn(
                        sim_probs, sim_vals)

                else:
                    sim_probs_plt, sim_vals_plt = sim_probs, sim_vals

                plt.plot(
                    sim_vals_plt,
                    sim_probs_plt,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            if lag_step in lag_steps_opt:
                suff = 'opt'

            else:
                suff = 'vld'

            plt.ylabel('Probability')
            plt.xlabel(f'{x_ax_label} (lag step(s) = {lag_step}_{suff})')

            out_name = f'{out_name_pref}_{data_label}_{lag_step:03d}.png'

            plt.savefig(
                str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site {var_label} CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_ts_probs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ts_probs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__ts_probs'

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:
            ref_ts_probs = h5_hdl[
                f'data_ref_rltzn/_ref_probs'][:, data_lab_idx]

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.plot(
                ref_ts_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_ts_probs = sim_grp_main[
                    f'{rltzn_lab}/probs'][:, data_lab_idx]

                plt.plot(
                    sim_ts_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'Time step')

            plt.ylim(0, 1)

            fig_name = f'{out_name_pref}_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site probability time series '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_phs_cdfs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_phs_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            ref_phs = np.sort(np.angle(h5_hdl[
                f'data_ref_rltzn/_ref_ft'][:, data_lab_idx]))

            ref_probs = np.arange(
                1.0, ref_phs.size + 1) / (ref_phs.size + 1.0)

#             ref_phs_dens_y = (ref_phs[1:] - ref_phs[:-1])

#             ref_phs_dens_x = ref_phs[:-1] + (0.5 * (ref_phs_dens_y))

            prob_pln_fig = plt.figure()
#             dens_plr_fig = plt.figure()
#             dens_pln_fig = plt.figure()

            plt.figure(prob_pln_fig.number)
            plt.plot(
                ref_phs,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

#             plt.figure(dens_plr_fig.number)
#             plt.polar(
#                 ref_phs_dens_x,
#                 ref_phs_dens_y,
#                 alpha=plt_sett.alpha_2,
#                 color=plt_sett.lc_2,
#                 lw=plt_sett.lw_1,
#                 label='ref')
#
#             plt.figure(dens_pln_fig.number)
#             plt.plot(
#                 ref_phs_dens_x,
#                 ref_phs_dens_y,
#                 alpha=plt_sett.alpha_2,
#                 color=plt_sett.lc_2,
#                 lw=plt_sett.lw_2,
#                 label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_phs = np.sort(
                    np.angle(sim_grp_main[f'{rltzn_lab}/ft'][:, data_lab_idx]))

#                 sim_phs_dens_y = (
#                     (sim_phs[1:] - sim_phs[:-1]) *
#                     ((sim_phs.size + 1) / (ref_phs.size + 1)))

                sim_probs = rankdata(sim_phs) / (sim_phs.size + 1.0)

#                 sim_phs_dens_x = sim_phs[:-1] + (0.5 * (sim_phs_dens_y))

                plt.figure(prob_pln_fig.number)
                plt.plot(
                    sim_phs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

#                 plt.figure(dens_plr_fig.number)
#                 plt.polar(
#                     sim_phs_dens_x,
#                     sim_phs_dens_y,
#                     alpha=plt_sett.alpha_1,
#                     color=plt_sett.lc_1,
#                     lw=plt_sett.lw_1,
#                     label=label)
#
#                 plt.figure(dens_pln_fig.number)
#                 plt.plot(
#                     sim_phs_dens_x,
#                     sim_phs_dens_y,
#                     alpha=plt_sett.alpha_1,
#                     color=plt_sett.lc_1,
#                     lw=plt_sett.lw_1,
#                     label=label)

                leg_flag = False

            # probs plain
            plt.figure(prob_pln_fig.number)

            plt.grid(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'FT Phase')

            fig_name = f'ss__phs_cdfs_plain_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

#             # dens polar
#             plt.figure(dens_plr_fig.number)
#
#             plt.grid(True)
#
#             plt.legend(framealpha=0.7)
#
#             plt.ylabel('Density\n\n')
#
#             plt.xlabel(f'FT Phase')
#
#             fig_name = (
#                 f'ss__phs_pdfs_polar_{data_labels[data_lab_idx]}_'
#                 f'{phs_cls_ctr}.png')
#
#             plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')
#
#             plt.close()

#             # dens plain
#             plt.figure(dens_pln_fig.number)
#
#             plt.grid(True)
#
#             plt.legend(framealpha=0.7)
#
#             plt.ylabel('Density')
#
#             plt.xlabel(f'FT Phase')
#
#             fig_name = (
#                 f'ss_phs_pdfs_plain_{data_labels[data_lab_idx]}_'
#                 f'{phs_cls_ctr}.png')
#
#             plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')
#
#             plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site phase spectrum CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_mag_cos_sin_cdfs_base(self, sin_cos_ftn, shrt_lab, lng_lab):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        assert sin_cos_ftn in [np.cos, np.sin], (
            'np.cos and np.sin allowed only!')

        plt_sett = self._plt_sett_mag_cos_sin_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        for data_lab_idx in loop_prod:

            ref_ft = h5_hdl[f'data_ref_rltzn/_ref_ft'][:, data_lab_idx]

            ref_phs = np.angle(ref_ft)

            ref_mag = np.abs(ref_ft)

            if shrt_lab == 'cos':
                ref_mag_cos_abs_ft = np.zeros_like(ref_mag, dtype=complex)
                ref_mag_cos_abs_ft.real = ref_mag * sin_cos_ftn(ref_phs)

                ref_mag_cos_abs = np.fft.irfft(ref_mag_cos_abs_ft)

            elif shrt_lab == 'sin':
                ref_mag_cos_abs_ft = np.zeros_like(ref_mag, dtype=complex)
                ref_mag_cos_abs_ft.imag = ref_mag * sin_cos_ftn(ref_phs)

                ref_mag_cos_abs = np.fft.irfft(ref_mag_cos_abs_ft)

            else:
                raise ValueError(f'Unknown shrt_lab: {shrt_lab}!')

            ref_mag_cos_abs.sort()

            ref_probs = np.arange(1.0, ref_mag_cos_abs.size + 1) / (
                (ref_mag_cos_abs.size + 1))

            plt.figure()

            plt.plot(
                ref_mag_cos_abs,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_probs = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_ft = sim_grp_main[f'{rltzn_lab}/ft'][:, data_lab_idx]

                sim_phs = np.angle(sim_ft)

                sim_mag = np.abs(sim_ft)

                if shrt_lab == 'cos':
                    sim_mag_cos_abs_ft = np.zeros_like(sim_mag, dtype=complex)
                    sim_mag_cos_abs_ft.real = sim_mag * sin_cos_ftn(sim_phs)

                    sim_mag_cos_abs = np.fft.irfft(sim_mag_cos_abs_ft)

                elif shrt_lab == 'sin':
                    sim_mag_cos_abs_ft = np.zeros_like(sim_mag, dtype=complex)
                    sim_mag_cos_abs_ft.imag = sim_mag * sin_cos_ftn(sim_phs)

                    sim_mag_cos_abs = np.fft.irfft(sim_mag_cos_abs_ft)

                else:
                    raise ValueError(f'Unknown shrt_lab: {shrt_lab}!')

                sim_mag_cos_abs.sort()

                if sim_probs is None:
                    sim_probs = np.arange(
                        1.0, sim_mag_cos_abs.size + 1.0) / (
                        (sim_mag_cos_abs.size + 1))

                plt.plot(
                    sim_mag_cos_abs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'iFT {lng_lab} value')

            fig_name = (
                f'ss__ift_{shrt_lab}_cdfs_{data_labels[data_lab_idx]}.png')

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site iFT {lng_lab} CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_mag_cdfs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_mag_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            ref_mag_abs = np.sort(np.abs(
                h5_hdl[f'data_ref_rltzn/_ref_ft'][:, data_lab_idx]))

            ref_probs = (
                np.arange(1.0, ref_mag_abs.size + 1) / (ref_mag_abs.size + 1))

            plt.figure()

            plt.plot(
                ref_mag_abs,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_mag_abs = np.sort(np.abs(
                    sim_grp_main[f'{rltzn_lab}/ft'][:, data_lab_idx]))

                sim_probs = rankdata(sim_mag_abs) / (sim_mag_abs.size + 1)

                plt.plot(
                    sim_mag_abs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'FT magnitude')

            fig_name = f'ss__mag_cdfs_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site magnitude spectrum CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_1D_vars(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_1D_vars_wider

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld'][:]
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps'][:]
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords_vld'][:]
        nth_ords_opt = h5_hdl['settings/_sett_obj_nth_ords'][:]

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        opt_idxs_steps = []
        for i, lag_step in enumerate(lag_steps):
            if lag_step not in lag_steps_opt:
                continue

            opt_idxs_steps.append((i, lag_step))

        opt_idxs_steps = np.array(opt_idxs_steps)

        opt_idxs_ords = []
        for i, nth_ord in enumerate(nth_ords):
            if nth_ord not in nth_ords_opt:
                continue

            opt_idxs_ords.append((i, nth_ord))

        opt_idxs_ords = np.array(opt_idxs_ords)

        opt_scatt_size_scale = 10

        for data_lab_idx in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            axes = plt.subplots(2, 3, squeeze=False)[1]

            axes[0, 0].plot(
                lag_steps,
                ref_grp['_ref_scorrs'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[0, 0].scatter(
                opt_idxs_steps[:, 1],
                ref_grp['_ref_scorrs'][data_lab_idx, opt_idxs_steps[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            axes[1, 0].plot(
                lag_steps,
                ref_grp['_ref_asymms_1'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[1, 0].scatter(
                opt_idxs_steps[:, 1],
                ref_grp['_ref_asymms_1'][data_lab_idx, opt_idxs_steps[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            axes[1, 1].plot(
                lag_steps,
                ref_grp['_ref_asymms_2'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[1, 1].scatter(
                opt_idxs_steps[:, 1],
                ref_grp['_ref_asymms_2'][data_lab_idx, opt_idxs_steps[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            axes[0, 1].plot(
                lag_steps,
                ref_grp['_ref_ecop_etpy'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[0, 1].scatter(
                opt_idxs_steps[:, 1],
                ref_grp['_ref_ecop_etpy'][data_lab_idx, opt_idxs_steps[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            axes[0, 2].plot(
                lag_steps,
                ref_grp['_ref_pcorrs'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[0, 2].scatter(
                opt_idxs_steps[:, 1],
                ref_grp['_ref_pcorrs'][data_lab_idx, opt_idxs_steps[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            axes[1, 2].plot(
                nth_ords,
                ref_grp['_ref_nths'][data_lab_idx,:],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[1, 2].scatter(
                opt_idxs_ords[:, 1],
                ref_grp['_ref_nths'][data_lab_idx, opt_idxs_ords[:, 0]],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                s=plt_sett.lw_2 * opt_scatt_size_scale)

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_grp = sim_grp_main[f'{rltzn_lab}']

                axes[0, 0].plot(
                    lag_steps,
                    sim_grp['scorrs'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 0].plot(
                    lag_steps,
                    sim_grp['asymms_1'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 1].plot(
                    lag_steps,
                    sim_grp['asymms_2'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[0, 1].plot(
                    lag_steps,
                    sim_grp['ecop_entps'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[0, 2].plot(
                    lag_steps,
                    sim_grp['pcorrs'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 2].plot(
                    nth_ords,
                    sim_grp['nths'][data_lab_idx,:],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            axes[0, 0].grid()
            axes[1, 0].grid()
            axes[1, 1].grid()
            axes[0, 1].grid()
            axes[0, 2].grid()
            axes[1, 2].grid()

            axes[0, 0].legend(framealpha=0.7)
#             axes[1, 0].legend(framealpha=0.7)
#             axes[1, 1].legend(framealpha=0.7)
#             axes[0, 1].legend(framealpha=0.7)
#             axes[0, 2].legend(framealpha=0.7)
#             axes[1, 2].legend(framealpha=0.7)

            axes[0, 0].set_ylabel('Spearman correlation')

            axes[1, 0].set_xlabel('Lag steps')
            axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

            axes[1, 1].set_xlabel('Lag steps')
            axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

            axes[0, 1].set_ylabel('Entropy')

            axes[0, 2].set_xlabel('Lag steps')
            axes[0, 2].set_ylabel('Pearson correlation')

            axes[1, 2].set_xlabel('Nth orders')
            axes[1, 2].set_ylabel('Dist. Sum')

            plt.tight_layout()

            fig_name = f'ss__summary_{data_labels[data_lab_idx]}.png'

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site optimized 1D objective function '
                f'variables took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cmpr_ecop_denss_base(args):

        (lag_steps,
         fig_suff,
         vmin,
         vmax,
         ecop_denss,
         cmap_mappable_beta,
         out_dir,
         plt_sett) = args

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        fig, axes = plt.subplots(rows, cols, squeeze=False)

        dx = 1.0 / (ecop_denss.shape[2] + 1.0)
        dy = 1.0 / (ecop_denss.shape[1] + 1.0)

        y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

        ax_ctr = 0
        row = 0
        col = 0
        for i in range(rows * cols):

            if i >= (lag_steps.size):
                axes[row, col].set_axis_off()

            else:
                axes[row, col].pcolormesh(
                    x,
                    y,
                    ecop_denss[i],
                    vmin=vmin,
                    vmax=vmax,
                    alpha=plt_sett.alpha_1,
                    shading='auto')

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.1,
                    0.85,
                    f'{lag_steps[i]} step(s) lag',
                    alpha=plt_sett.alpha_2)

                if col:
                    axes[row, col].set_yticklabels([])

                else:
                    axes[row, col].set_ylabel('Probability (lagged)')

                if row < (rows - 1):
                    axes[row, col].set_xticklabels([])

                else:
                    axes[row, col].set_xlabel('Probability')

            col += 1
            if not (col % cols):
                row += 1
                col = 0

            ax_ctr += 1

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Empirical copula density',
            extend='max',
            alpha=plt_sett.alpha_1,
            drawedges=False)

        plt.savefig(
            str(out_dir / f'ss__ecop_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_denss(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ecops_denss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld'][:]
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            fig_suff = f'ref_{data_labels[data_lab_idx]}'

            ecop_denss = h5_hdl[
                f'data_ref_rltzn/_ref_ecop_dens'][data_lab_idx,:,:,:]

            vmin = 0.0
#             vmax = ecop_denss.mean() * 2.0
            vmax = ecop_denss.max() * 0.85

            cmap_mappable_beta = plt.cm.ScalarMappable(
                norm=Normalize(vmin / 100, vmax / 100, clip=True),
                cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            args = (
                lag_steps,
                fig_suff,
                vmin,
                vmax,
                ecop_denss,
                cmap_mappable_beta,
                self._ss_dir,
                plt_sett)

            self._plot_cmpr_ecop_denss_base(args)

            plot_ctr = 0
            for rltzn_lab in sim_grp_main:
                fig_suff = (
                    f'sim_{data_labels[data_lab_idx]}_{rltzn_lab}')

                ecop_denss = sim_grp_main[
                    f'{rltzn_lab}/ecop_dens'][data_lab_idx,:,:,:]

                args = (
                    lag_steps,
                    fig_suff,
                    vmin,
                    vmax,
                    ecop_denss,
                    cmap_mappable_beta,
                    self._ss_dir,
                    plt_sett)

                self._plot_cmpr_ecop_denss_base(args)

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site ecop densities '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cmpr_ecop_scatter_base(args):

        (lag_steps,
         fig_suff,
         probs,
         out_dir,
         plt_sett,
         cmap_mappable_beta,
         clrs) = args

        rows = int(ceil(lag_steps.size ** 0.5))
        cols = ceil(lag_steps.size / rows)

        fig, axes = plt.subplots(rows, cols, squeeze=False)

        row = 0
        col = 0
        for i in range(rows * cols):

            if i >= (lag_steps.size):
                axes[row, col].set_axis_off()

            else:
                probs_i, rolled_probs_i = roll_real_2arrs(
                    probs, probs, lag_steps[i], True)

                axes[row, col].scatter(
                    probs_i,
                    rolled_probs_i,
                    c=clrs[:-lag_steps[i]],
                    alpha=plt_sett.alpha_1)

                axes[row, col].grid()

                axes[row, col].set_aspect('equal')

                axes[row, col].text(
                    0.05,
                    0.9,
                    f'{lag_steps[i]} step(s) lag',
                    alpha=plt_sett.alpha_2)

                if col:
                    axes[row, col].set_yticklabels([])

                else:
                    axes[row, col].set_ylabel('Probability (lagged)')

                if row < (rows - 1):
                    axes[row, col].set_xticklabels([])

                else:
                    axes[row, col].set_xlabel('Probability')

            col += 1
            if not (col % cols):
                row += 1
                col = 0

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Timing',
            drawedges=False)

        plt.savefig(
            str(out_dir / f'ss__ecops_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_scatter(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_str = 'jet'

        cmap_beta = plt.get_cmap(cmap_str)

        for data_lab_idx in loop_prod:

            probs = h5_hdl[f'data_ref_rltzn/_ref_probs'][:, data_lab_idx]

            fig_suff = f'ref_{data_labels[data_lab_idx]}'

            cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            ref_timing_ser = np.arange(
                1.0, probs.size + 1.0) / (probs.size + 1.0)

            ref_clrs = plt.get_cmap(cmap_str)(ref_timing_ser)

            args = (
                lag_steps,
                fig_suff,
                probs,
                self._ss_dir,
                plt_sett,
                cmap_mappable_beta,
                ref_clrs)

            self._plot_cmpr_ecop_scatter_base(args)

            plot_ctr = 0
            sim_timing_ser = None
            for rltzn_lab in sim_grp_main:
                probs = sim_grp_main[f'{rltzn_lab}/probs'][:, data_lab_idx]

                if sim_timing_ser is None:
                    sim_timing_ser = np.arange(
                        1.0, probs.size + 1.0) / (probs.size + 1.0)

                    sim_clrs = plt.get_cmap(cmap_str)(sim_timing_ser)

                fig_suff = f'sim_{data_labels[data_lab_idx]}_{rltzn_lab}'

                args = (
                    lag_steps,
                    fig_suff,
                    probs,
                    self._ss_dir,
                    plt_sett,
                    cmap_mappable_beta,
                    sim_clrs)

                self._plot_cmpr_ecop_scatter_base(args)

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site ecop scatters '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_nth_ord_diffs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_nth_ord_diffs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'ss__nth_diff_cdfs'

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords_vld'][:]
        nth_ords_opt = h5_hdl['settings/_sett_obj_nth_ords'][:]
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, nth_ords)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, nth_ord) in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_probs = ref_grp['_ref_nth_ord_diffs_cdfs_'
                f'dict_{data_label}_{nth_ord:03d}_y'][:]

            ref_vals = ref_grp[
                f'_ref_nth_ord_diffs_cdfs_dict_{data_label}_{nth_ord:03d}_x'][:]

            if self._dens_dist_flag:
                ref_probs_plt, ref_vals_plt = self._get_dens_ftn(
                    ref_probs, ref_vals)

            else:
                ref_probs_plt, ref_vals_plt = ref_probs, ref_vals

            plt.figure()

            plt.plot(
                ref_vals_plt,
                ref_probs_plt,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            sim_probs = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/nth_ord_'
                    f'diffs_{data_label}_{nth_ord:03d}'][:]

                if sim_probs is None:
                    sim_probs = np.arange(
                        1.0, sim_vals.size + 1.0) / (sim_vals.size + 1)

                if self._dens_dist_flag:
                    sim_probs_plt, sim_vals_plt = self._get_dens_ftn(
                        sim_probs, sim_vals)

                else:
                    sim_probs_plt, sim_vals_plt = sim_probs, sim_vals

                plt.plot(
                    sim_vals_plt,
                    sim_probs_plt,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            if nth_ord in nth_ords_opt:
                suff = 'opt'

            else:
                suff = 'vld'

            plt.xlabel(f'Difference (order = {nth_ord}_{suff})')

            out_name = f'{out_name_pref}_{data_label}_{nth_ord:03d}.png'

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site nth-order differences '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cmpr_ft_corrs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_cumm_corrs = ref_grp['_ref_ft_cumm_corr'][:, data_lab_idx]

            ref_periods = ((ref_cumm_corrs.size * 2) + 2) / (
                np.arange(1, ref_cumm_corrs.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_cumm_corrs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref-ref')

            sim_periods = None
            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim-ref'

                else:
                    label = None

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/ft_cumm_corr_sim_ref'][:, data_lab_idx]

                if sim_periods is None:
                    sim_periods = ((sim_cumm_corrs.size * 2) + 2) / (
                        np.arange(1, sim_cumm_corrs.size + 1))

                plt.semilogx(
                    sim_periods,
                    sim_cumm_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative correlation')

            plt.xlabel('Period (steps)')

            plt.ylim(-1, +1)

            plt.xlim(plt.xlim()[::-1])

            out_name = (
                f'ss__ft_cumm_corrs_sim_ref_'
                f'{data_labels[data_lab_idx]}.png')

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

            # cumm ft corrs, sim_ref, xy
            plt.figure()

            for rltzn_lab in sim_grp_main:

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/ft_cumm_corr_sim_ref'][:, data_lab_idx]

                plt.plot(
                    ref_cumm_corrs,
                    sim_cumm_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.grid()

            plt.ylabel('Simulation-Reference cummulative correlation')
            plt.xlabel(f'Reference-Reference cummulative correlation')

            plt.xlim(+0, +1)
            plt.ylim(-1, +1)

            plt.gca().set_aspect('equal', 'box')

            out_name = (
                f'ss__ft_cumm_corrs_xy_sim_ref_'
                f'{data_labels[data_lab_idx]}.png')

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

            # cumm ft corrs, sim_sim
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_cumm_corrs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref-ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim-sim'

                else:
                    label = None

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/ft_cumm_corr_sim_sim'][:, data_lab_idx]

                plt.semilogx(
                    sim_periods,
                    sim_cumm_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative correlation')
            plt.xlabel('Period (steps)')

            plt.ylim(-1, +1)

            plt.xlim(plt.xlim()[::-1])

            out_name = (
                f'ss__ft_cumm_corrs_sim_sim_'
                f'{data_labels[data_lab_idx]}.png')

            plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

            # diff cumm ft corrs
            plt.figure()

            ref_freq_corrs = np.concatenate((
                [ref_cumm_corrs[0]],
                 ref_cumm_corrs[1:] - ref_cumm_corrs[:-1]))

            plt.semilogx(
                ref_periods,
                ref_freq_corrs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref-ref')

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim-ref'

                else:
                    label = None

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/ft_cumm_corr_sim_ref'][:, data_lab_idx]

                sim_freq_corrs = np.concatenate((
                    [sim_cumm_corrs[0]],
                    sim_cumm_corrs[1:] - sim_cumm_corrs[:-1]))

                plt.plot(
                    sim_periods,
                    sim_freq_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Differential correlation')

            plt.xlabel('Period (steps)')

            max_ylim = max(np.abs(plt.ylim()))

            plt.ylim(-max_ylim, +max_ylim)

            plt.xlim(plt.xlim()[::-1])

            out_name = (
                f'ss__ft_diff_freq_corrs_sim_ref_'
                f'{data_labels[data_lab_idx]}.png')

            plt.savefig(
                str(self._ss_dir / out_name), bbox_inches='tight')

            plt.close()

#             # cumm phs corrs, sim_sim
#             plt.figure()
#
#             ref_phs_spec = ref_grp['_ref_phs_spec'][:, data_lab_idx][1:-1]
#
#             leg_flag = True
#             for rltzn_lab in sim_grp_main:
#                 if leg_flag:
#                     label = 'sim-sim'
#
#                 else:
#                     label = None
#
#                 sim_phs_spec = sim_grp_main[
#                     f'{rltzn_lab}/phs_spec'][:, data_lab_idx][1:-1]
#
#                 sim_phs_corr = np.cumsum(
#                     np.cos(ref_phs_spec - sim_phs_spec)) / sim_phs_spec.size
#
#                 plt.semilogx(
#                     sim_periods,
#                     sim_phs_corr,
#                     alpha=plt_sett.alpha_1,
#                     color=plt_sett.lc_1,
#                     lw=plt_sett.lw_1,
#                     label=label)
#
#                 leg_flag = False
#
#             plt.grid()
#
#             plt.legend(framealpha=0.7)
#
#             plt.ylabel('Cummulative correlation')
#             plt.xlabel('Period (steps)')
#
#             plt.ylim(-1, +1)
#
#             plt.xlim(plt.xlim()[::-1])
#
#             out_name = (
#                 f'ss__phs_cumm_corrs_sim_sim_'
#                 f'{data_labels[data_lab_idx]}.png')
#
#             plt.savefig(str(self._ss_dir / out_name), bbox_inches='tight')
#
#             plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site FT correlations '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return
