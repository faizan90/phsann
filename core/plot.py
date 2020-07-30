'''
@author: Faizan-Uni

Jan 16, 2020

1:32:31 PM
'''
import psutil
import matplotlib as mpl
from multiprocessing import Pool
from timeit import default_timer
from itertools import product, combinations

# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from math import ceil
from pathlib import Path

import h5py
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.interpolate import interp1d
from matplotlib.colors import Normalize

from ..cyth import fill_bi_var_cop_dens
from ..misc import print_sl, print_el, roll_real_2arrs

plt.ioff()


class PlotSettings:

    '''For internal use

    Description of settings at:
    https://matplotlib.org/tutorials/introductory/customizing.html
    '''

    def __init__(self, figsize, dpi, fontsize):

        assert isinstance(figsize, tuple), 'figsize not a tuple!'

        assert len(figsize) == 2, 'Only two items allowed in figsize!'

        assert all([isinstance(size, (int, float)) for size in figsize]), (
            'Items in figsize can either be integers or floats!')

        assert all([size > 0 for size in figsize]), (
            'items in figsize cannot be <= 0!')

        assert isinstance(dpi, int), 'dpi not an integer!'

        assert dpi > 0, 'dpi should be > 0!'

        assert isinstance(fontsize, int), 'fontsize not an integer!'

        assert fontsize > 0

        self.prms_dict = {
            'figure.figsize': figsize,
            'figure.dpi': dpi,
            'font.size': fontsize,
            }

        return


class PlotLineSettings(PlotSettings):

    '''For internal use'''

    def __init__(
            self,
            figsize,
            dpi,
            fontsize,
            alpha_1,
            alpha_2,
            lw_1,
            lw_2,
            lc_1,
            lc_2):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha can only be 0 <= alpha_2 <= 1!'

        assert isinstance(lw_1, (int, float)), (
            'lw_1 not an integer or a float!')

        assert lw_1 > 0, 'lw_1 must be > 0!'

        assert isinstance(lw_2, (int, float)), (
            'lw_2 not an integer or a float!')

        assert lw_2 > 0, 'lw_2 must be > 0!'

        assert isinstance(lc_1, (str, tuple, hex)), 'Invalid lc_1!'
        assert isinstance(lc_2, (str, tuple, hex)), 'Invalid lc_2!'

        self.prms_dict.update({
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2

        self.lw_1 = lw_1
        self.lw_2 = lw_2

        self.lc_1 = lc_1
        self.lc_2 = lc_2
        return


class PlotImageSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, cmap):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(cmap, str), 'cmap not a string!'

        assert cmap in mpl_cm.cmap_d, 'Unknown cmap!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            'image.cmap': cmap,
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        return


class PlotScatterSettings(PlotSettings):

    '''For internal use'''

    def __init__(self, figsize, dpi, fontsize, alpha_1, alpha_2, color):

        PlotSettings.__init__(self, figsize, dpi, fontsize)

        assert isinstance(color, (str, tuple, hex)), 'Invalid color!'

        assert isinstance(alpha_1, (int, float)), (
            'alpha_1 can only be an integer or a float!')

        assert 0 <= alpha_1 <= 1, 'alpha_text can only be 0 <= alpha_1 <= 1!'

        assert isinstance(alpha_2, (int, float)), (
            'alpha_2 can only be an integer or a float!')

        assert 0 <= alpha_2 <= 1, 'alpha_text can only be 0 <= alpha_2 <= 1!'

        self.prms_dict.update({
            })

        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.color = color
        return


def get_mpl_prms(keys):

    prms_dict = {key: plt.rcParams[key] for key in keys}

    return prms_dict


def set_mpl_prms(prms_dict):

    plt.rcParams.update(prms_dict)

    return


class PhaseAnealingPlotOSV:

    '''
    Optimization state variables' plots
    '''

    def _plot_tmrs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_tmrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        call_times_fig = plt.figure()
        n_calls_fig = plt.figure()

        for rltzn_lab in sim_grp_main:
            times_grp = sim_grp_main[f'{rltzn_lab}/tmr_call_times']

            n_calls_grp = sim_grp_main[f'{rltzn_lab}/tmr_n_calls']

            # Times
            plt.figure(call_times_fig.number)
            for i, lab in enumerate(times_grp.attrs):
                plt.bar(
                    i,
                    times_grp.attrs[lab],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1)

            # No. calls
            plt.figure(n_calls_fig.number)
            for i, lab in enumerate(n_calls_grp.attrs):
                plt.bar(
                    i,
                    n_calls_grp.attrs[lab],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1)

        # Times
        plt.figure(call_times_fig.number)
#             plt.yscale('log')
        plt.xlabel('Method')
        plt.ylabel(f'Cummulative time spent (sec)')
        plt.xticks(
            np.arange(len(times_grp.attrs)), times_grp.attrs, rotation=90)
        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__tmr_call_times.png'),
            bbox_inches='tight')

        plt.close()

        # No. calls
        plt.figure(n_calls_fig.number)
#             plt.yscale('log')
        plt.xlabel('Method')
        plt.ylabel(f'Cummulative calls')
        plt.xticks(
            np.arange(len(n_calls_grp.attrs)), n_calls_grp.attrs, rotation=90)
        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__tmr_n_calls.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization method timers '
                f'took {end_tm - beg_tm:0.2f} seconds.')

        return

    def _plot_phs_idxs_sclrs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            phs_idxs_sclrs_all = sim_grp_main[f'{rltzn_lab}/idxs_sclrs']

            plt.plot(
                phs_idxs_sclrs_all[:, 0],
                phs_idxs_sclrs_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase indices reduction rate')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__phs_idxs_sclrs.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization indices scaler rates '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_tols(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_tols

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        beg_iters = h5_hdl['settings'].attrs['_sett_ann_obj_tol_iters']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            tol_iters = beg_iters + np.arange(
                sim_grp_main[f'{rltzn_lab}/tols'].shape[0])

            plt.semilogy(
                tol_iters,
                sim_grp_main[f'{rltzn_lab}/tols'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean absolute objective function\ndifference of previous '
            f'{beg_iters} iterations')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__tols.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization tolerances '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_obj_vals_indiv(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_objs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        obj_flag_vals = h5_hdl['settings/_sett_obj_flag_vals'][...]
        obj_flag_labels = h5_hdl['settings/_sett_obj_flag_labels'][...]

        obj_flag_idx = 0
        for i, (obj_flag_val, obj_flag_label) in enumerate(
            zip(obj_flag_vals, obj_flag_labels)):

            if not obj_flag_val:
                continue

            plt.figure()
            for rltzn_lab in sim_grp_main:
                loc = f'{rltzn_lab}/obj_vals_all_indiv'

                plt.semilogy(
                    sim_grp_main[loc][:, obj_flag_idx],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.xlabel('Iteration')

            plt.ylabel(f'{obj_flag_label}\nobjective function value')

            plt.grid()

            fig_name = f'osv__obj_vals_all_indiv_{i:02d}.png'

            plt.savefig(
                str(self._osv_dir / fig_name), bbox_inches='tight')

            plt.close()

            obj_flag_idx += 1

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization individual objective function values '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_obj_vals(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_objs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # obj_vals_all
        plt.figure()
        for rltzn_lab in sim_grp_main:
            plt.semilogy(
                sim_grp_main[f'{rltzn_lab}/obj_vals_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Raw objective function value')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__obj_vals_all.png'),
            bbox_inches='tight')

        plt.close()

        # obj_vals_min
        plt.figure()
        for rltzn_lab in sim_grp_main:
            plt.semilogy(
                sim_grp_main[f'{rltzn_lab}/obj_vals_min'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Running minimum objective function value')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__obj_vals_min.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization cummulative objective function values '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_acpt_rates(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_acpt_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        acpt_rate_iters = (
            h5_hdl['settings'].attrs['_sett_ann_acpt_rate_iters'])

        # acpt_rates_all
        plt.figure()
        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/acpt_rates_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Running mean acceptance rate')

        plt.grid()

        plt.savefig(
            str(self._osv_dir /
                f'osv__acpt_rates.png'),
            bbox_inches='tight')

        plt.close()

        # acpt_rates_dfrntl
        plt.figure()
        for rltzn_lab in sim_grp_main:
            acpt_rate_dfrntl = sim_grp_main[f'{rltzn_lab}/acpt_rates_dfrntl']

            plt.plot(
                acpt_rate_dfrntl[:, 0],
                acpt_rate_dfrntl[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean acceptance rate for the\npast {acpt_rate_iters} '
            f'iterations')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__acpt_rates_dfrntl.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization acceptance rates '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_idxs(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_idxs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        idxs_all_hist_fig = plt.figure()
        idxs_acpt_hist_fig = plt.figure()
        idxs_acpt_rel_hist_fig = plt.figure()

        for rltzn_lab in sim_grp_main:
            idxs_all = sim_grp_main[f'{rltzn_lab}/idxs_all'][...]

            idxs_acpt = sim_grp_main[f'{rltzn_lab}/idxs_acpt'][...]

            rel_freqs = np.zeros_like(idxs_all, dtype=np.float64)

            non_zero_idxs = idxs_all.astype(bool)

            rel_freqs[non_zero_idxs] = (
                idxs_acpt[non_zero_idxs] / idxs_all[non_zero_idxs])

            freqs = np.arange(idxs_all.size)

            plt.figure(idxs_all_hist_fig.number)
            plt.bar(
                freqs,
                idxs_all,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

            plt.figure(idxs_acpt_hist_fig.number)
            plt.bar(
                freqs,
                idxs_acpt,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

            plt.figure(idxs_acpt_rel_hist_fig.number)
            plt.bar(
                freqs,
                rel_freqs,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1)

        # idxs_all
        plt.figure(idxs_all_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Raw frequency')
        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__idxs_all_hist.png'),
            bbox_inches='tight')

        plt.close()

        # idxs_acpt
        plt.figure(idxs_acpt_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Acceptance frequency')
        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__idxs_acpt_hist.png'),
            bbox_inches='tight')

        plt.close()

        # idxs_acpt_rel
        plt.figure(idxs_acpt_rel_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Relative acceptance frequency')
        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__idxs_acpt_rel_hist.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization frequency indices '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_temps(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_temps

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            temps_all = sim_grp_main[f'{rltzn_lab}/temps']

            plt.semilogy(
                temps_all[:, 0],
                temps_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Annealing temperature')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__temps.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization temperatures '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_phs_red_rates(self):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            phs_red_rates_all = sim_grp_main[f'{rltzn_lab}/phs_red_rates']

            plt.plot(
                phs_red_rates_all[:, 0],
                phs_red_rates_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase increment reduction rate')

        plt.grid()

        plt.savefig(
            str(self._osv_dir / f'osv__phs_red_rates.png'),
            bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting optimization reduction rates '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return


class PhaseAnnealingPlotSingleSite:

    '''
    Single-site plots
    '''

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

        finer_vals = np.linspace(vals[+0], vals[-1], vals.size * 2)

        interp_ftn = interp1d(
            vals,
            probs,
            kind='slinear')

        finer_probs = interp_ftn(
            (finer_vals + ((finer_vals[1] - finer_vals[0]) * 0.5))[:-1])

        finer_vals = finer_vals[1:-1]

        finer_probs_diff = finer_probs[1:] - finer_probs[:-1]

        dens = finer_probs_diff / (finer_vals[1] - finer_vals[0])
        dens[dens < 0] = 0
        dens /= dens.sum()

        assert dens.size == finer_vals.size

        return dens, finer_vals

    def _plot_gnrc_cdfs_cmpr(self, var_label, x_ax_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ss__{var_label}_diff_cdfs'

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, lag_steps)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, lag_step) in loop_prod:

            ref_probs = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{data_label}_{lag_step:03d}_y'][:]

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{data_label}_{lag_step:03d}_x']

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
                    f'diffs_{data_label}_{lag_step:03d}']

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

            ref_mag_cos_abs = np.sort(ref_mag * sin_cos_ftn(ref_phs))

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

                sim_mag_cos_abs = np.sort(sim_mag * sin_cos_ftn(sim_phs))

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

            plt.xlabel(f'FT {lng_lab} magnitude')

            fig_name = (
                f'ss__mag_{shrt_lab}_cdfs_{data_labels[data_lab_idx]}.png')

            plt.savefig(str(self._ss_dir / fig_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site FT {lng_lab} CDFs '
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

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        lag_steps_opt = h5_hdl['settings/_sett_obj_lag_steps']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords_vld']
        nth_ords_opt = h5_hdl['settings/_sett_obj_nth_ords']

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
                ref_grp['_ref_scorrs'][data_lab_idx, :],
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
                ref_grp['_ref_asymms_1'][data_lab_idx, :],
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
                ref_grp['_ref_asymms_2'][data_lab_idx, :],
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
                ref_grp['_ref_ecop_etpy'][data_lab_idx, :],
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
                ref_grp['_ref_pcorrs'][data_lab_idx, :],
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
                ref_grp['_ref_nths'][data_lab_idx, :],
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
                    sim_grp['scorrs'][data_lab_idx, :],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 0].plot(
                    lag_steps,
                    sim_grp['asymms_1'][data_lab_idx, :],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 1].plot(
                    lag_steps,
                    sim_grp['asymms_2'][data_lab_idx, :],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[0, 1].plot(
                    lag_steps,
                    sim_grp['ecop_entps'][data_lab_idx, :],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[0, 2].plot(
                    lag_steps,
                    sim_grp['pcorrs'][data_lab_idx, :],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 2].plot(
                    nth_ords,
                    sim_grp['nths'][data_lab_idx, :],
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
            axes[1, 0].legend(framealpha=0.7)
            axes[1, 1].legend(framealpha=0.7)
            axes[0, 1].legend(framealpha=0.7)
            axes[0, 2].legend(framealpha=0.7)
            axes[1, 2].legend(framealpha=0.7)

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
                    alpha=plt_sett.alpha_1)

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

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps_vld']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        loop_prod = np.arange(n_data_labels)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for data_lab_idx in loop_prod:

            fig_suff = f'ref_{data_labels[data_lab_idx]}'

            ecop_denss = h5_hdl[
                f'data_ref_rltzn/_ref_ecop_dens'][data_lab_idx, :, :, :]

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

            for rltzn_lab in sim_grp_main:
                fig_suff = (
                    f'sim_{data_labels[data_lab_idx]}_{rltzn_lab}')

                ecop_denss = sim_grp_main[
                    f'{rltzn_lab}/ecop_dens'][data_lab_idx, :, :, :]

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
                    probs, probs, lag_steps[i])

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

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords_vld']
        nth_ords_opt = h5_hdl['settings/_sett_obj_nth_ords']
        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        loop_prod = product(data_labels, nth_ords)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (data_label, nth_ord) in loop_prod:

            ref_grp = h5_hdl[f'data_ref_rltzn']

            ref_probs = ref_grp['_ref_nth_ord_diffs_cdfs_'
                f'dict_{data_label}_{nth_ord:03d}_y'][:]

            ref_vals = ref_grp[
                f'_ref_nth_ord_diffs_cdfs_dict_{data_label}_{nth_ord:03d}_x']

            plt.figure()

            plt.plot(
                ref_vals,
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

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/nth_ord_'
                    f'diffs_{data_label}_{nth_ord:03d}']

                if sim_probs is None:
                    sim_probs = np.arange(
                        1.0, sim_vals.size + 1.0) / (sim_vals.size + 1)

                plt.plot(
                    sim_vals,
                    sim_probs,
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

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site FT correlations '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return


class PhaseAnnealingPlotMultiSite:

    def _plot_cross_gnrc_cdfs(self, var_label, x_ax_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ms__cross_{var_label}_diff_cdfs'

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        combs = combinations(data_labels, 2)

        loop_prod = combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for cols in loop_prod:

            assert len(cols) == 2

            ref_probs = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_y'][:]

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_diffs_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_x']

            plt.figure()

            plt.plot(
                ref_vals,
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

                sim_vals = sim_grp_main[
                    f'{rltzn_lab}/{var_label}_diffs_{cols[0]}_{cols[1]}']

                sim_probs = rankdata(sim_vals) / (sim_vals.size + 1)

                plt.plot(
                    sim_vals,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')
            plt.xlabel(f'{x_ax_label}')

            out_name = f'{out_name_pref}_{"_".join(cols)}.png'

            plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site {var_label} CDFs '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_denss_cntmnt(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_denss_cntmnt

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_ecop_dens_bins = h5_hdl['settings'].attrs['_sett_obj_ecop_dens_bins']

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_beta = plt.get_cmap('Accent')._resample(3)  # plt.get_cmap(plt.rcParams['image.cmap'])

        cmap_beta.colors[1, :] = [1, 1, 1, 1]

        ref_ecop_dens_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            np.nan,
            dtype=np.float64)

        sim_ecop_dens_mins_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            +np.inf,
            dtype=np.float64)

        sim_ecop_dens_maxs_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            -np.inf,
            dtype=np.float64)

        tem_ecop_dens_arr = np.empty_like(ref_ecop_dens_arr)

        cntmnt_ecop_dens_arr = np.empty_like(ref_ecop_dens_arr)

        cmap_mappable_beta = plt.cm.ScalarMappable(
            norm=Normalize(-1, +1, clip=True), cmap=cmap_beta)

        cmap_mappable_beta.set_array([])

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/_ref_probs'][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/_ref_probs'][:, di_b]

            fill_bi_var_cop_dens(probs_a, probs_b, ref_ecop_dens_arr)

            for rltzn_lab in sim_grp_main:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'][:, di_b]

                fill_bi_var_cop_dens(probs_a, probs_b, tem_ecop_dens_arr)

                sim_ecop_dens_mins_arr = np.minimum(
                    sim_ecop_dens_mins_arr, tem_ecop_dens_arr)

                sim_ecop_dens_maxs_arr = np.maximum(
                    sim_ecop_dens_maxs_arr, tem_ecop_dens_arr)

            cntmnt_ecop_dens_arr[:] = 0.0
            cntmnt_ecop_dens_arr[ref_ecop_dens_arr < sim_ecop_dens_mins_arr] = -1
            cntmnt_ecop_dens_arr[ref_ecop_dens_arr > sim_ecop_dens_maxs_arr] = +1

            fig_suff = f'ms__cross_ecop_dens_cnmnt_{dl_a}_{dl_b}'

            fig, axes = plt.subplots(1, 1, squeeze=False)

            row, col = 0, 0

            dx = 1.0 / (cntmnt_ecop_dens_arr.shape[1] + 1.0)
            dy = 1.0 / (cntmnt_ecop_dens_arr.shape[0] + 1.0)

            y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

            axes[row, col].pcolormesh(
                x,
                y,
                cntmnt_ecop_dens_arr,
                vmin=-1,
                vmax=+1,
                alpha=plt_sett.alpha_1,
                cmap=cmap_beta)

            axes[row, col].set_aspect('equal')

            axes[row, col].set_ylabel('Probability')
            axes[row, col].set_xlabel('Probability')

            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_ylim(0, 1)

            cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

            cb = plt.colorbar(
                mappable=cmap_mappable_beta,
                cax=cbaxes,
                orientation='horizontal',
                label='Empirical copula density containment',
                alpha=plt_sett.alpha_1,
                ticks=[-1, 0, +1],
                drawedges=False)

            cb.ax.set_xticklabels(['Too hi.', 'Within', 'Too lo.'])

            plt.savefig(
                str(self._ms_dir / f'ms__cross_ecops_denss_cmpr_{fig_suff}.png'),
                bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop density containment '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_denss(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_denss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        n_ecop_dens_bins = h5_hdl['settings'].attrs['_sett_obj_ecop_dens_bins']

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        ecop_dens_arr = np.full(
            (n_ecop_dens_bins, n_ecop_dens_bins),
            np.nan,
            dtype=np.float64)

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/_ref_probs'
                ][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/_ref_probs'
                ][:, di_b]

            fill_bi_var_cop_dens(probs_a, probs_b, ecop_dens_arr)

            fig_suff = f'ref_{dl_a}_{dl_b}'

            vmin = 0.0
#             vmax = np.mean(ecop_dens_arr) * 2.0
            vmax = np.max(ecop_dens_arr) * 0.85

            cmap_mappable_beta = plt.cm.ScalarMappable(
                norm=Normalize(vmin / 100, vmax / 100, clip=True),
                cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            args = (
                fig_suff,
                vmin,
                vmax,
                ecop_dens_arr,
                cmap_mappable_beta,
                self._ms_dir,
                plt_sett)

            self._plot_cross_ecop_denss_base(args)

            for rltzn_lab in sim_grp_main:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'
                    ][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'
                    ][:, di_b]

                fill_bi_var_cop_dens(probs_a, probs_b, ecop_dens_arr)

                fig_suff = f'sim_{dl_a}_{dl_b}_{rltzn_lab}'

                args = (
                    fig_suff,
                    vmin,
                    vmax,
                    ecop_dens_arr,
                    cmap_mappable_beta,
                    self._ms_dir,
                    plt_sett)

                self._plot_cross_ecop_denss_base(args)

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop densities '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cross_ecop_denss_base(args):

        (fig_suff,
         vmin,
         vmax,
         ecop_dens_arr,
         cmap_mappable_beta,
         out_dir,
         plt_sett) = args

        fig, axes = plt.subplots(1, 1, squeeze=False)

        row, col = 0, 0

        dx = 1.0 / (ecop_dens_arr.shape[1] + 1.0)
        dy = 1.0 / (ecop_dens_arr.shape[0] + 1.0)

        y, x = np.mgrid[slice(dy, 1.0, dy), slice(dx, 1.0, dx)]

        axes[row, col].pcolormesh(
            x,
            y,
            ecop_dens_arr,
            vmin=vmin,
            vmax=vmax,
            alpha=plt_sett.alpha_1)

        axes[row, col].set_aspect('equal')

        axes[row, col].set_ylabel('Probability')
        axes[row, col].set_xlabel('Probability')

        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)

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
            str(out_dir / f'ms__cross_ecops_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    @staticmethod
    def _get_cross_ft_cumm_corr(mag_arr, phs_arr):

        mags_prod = np.prod(mag_arr, axis=1)

        min_phs = phs_arr.min(axis=1)
        max_phs = phs_arr.max(axis=1)

        ft_env_cov = mags_prod * np.cos(max_phs - min_phs)

        mag_sq_sum = np.prod((mag_arr ** 2).sum(axis=0)) ** 0.5

        assert mag_sq_sum > 0

        ft_corr = np.cumsum(ft_env_cov) / mag_sq_sum

        assert np.isfinite(ft_corr[-1])
        return ft_corr

    def _plot_cross_ft_corrs(self):

        '''
        Meant for pairs right now.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        # It can be done for all posible combinations by having a loop here.
        data_label_combs = combinations(data_labels, 2)

        loop_prod = data_label_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for (dl_a, dl_b) in loop_prod:

            ref_ft_cumm_corr = self._get_cross_ft_cumm_corr(
                h5_hdl[f'data_ref_rltzn/_ref_mag_spec'][...],
                h5_hdl[f'data_ref_rltzn/_ref_phs_spec'][...])

#             ref_freqs = np.arange(1, ref_ft_cumm_corr.size + 1)

            ref_periods = (ref_ft_cumm_corr.size * 2) / (
                np.arange(1, ref_ft_cumm_corr.size + 1))

            # cumm ft corrs, sim_ref
            plt.figure()

            plt.semilogx(
                ref_periods,
                ref_ft_cumm_corr,
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

                sim_ft_cumm_corr = self._get_cross_ft_cumm_corr(
                    sim_grp_main[f'{rltzn_lab}/mag_spec'][...],
                    sim_grp_main[f'{rltzn_lab}/phs_spec'][...])

                if sim_periods is None:
                    sim_periods = np.arange(1, sim_ft_cumm_corr.size + 1)

                plt.semilogx(
                    sim_periods,
                    sim_ft_cumm_corr,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Cummulative correlation')

            plt.xlabel(f'Period (steps)')

            plt.xlim(plt.xlim()[::-1])

            plt.ylim(-1, +1)

            out_name = f'ms__ft_cross_cumm_corrs_{dl_a}_{dl_b}.png'

            plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross FT correlations '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_ecop_scatter(self):

        '''
        Meant for pairs only.
        '''

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_cross_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        data_label_idx_combs = combinations(enumerate(data_labels), 2)

        loop_prod = data_label_idx_combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_str = 'jet'

        cmap_beta = plt.get_cmap(cmap_str)

        for ((di_a, dl_a), (di_b, dl_b)) in loop_prod:

            probs_a = h5_hdl[f'data_ref_rltzn/_ref_probs'][:, di_a]

            probs_b = h5_hdl[f'data_ref_rltzn/_ref_probs'][:, di_b]

            fig_suff = f'ref_{dl_a}_{dl_b}'

            cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            ref_timing_ser = np.arange(
                1.0, probs_a.size + 1.0) / (probs_a.size + 1.0)

            ref_clrs = plt.get_cmap(cmap_str)(ref_timing_ser)

            sim_timing_ser = ref_timing_ser
            sim_clrs = ref_clrs

            args = (
                probs_a,
                probs_b,
                fig_suff,
                self._ms_dir,
                plt_sett,
                cmap_mappable_beta,
                ref_clrs)

            self._plot_cross_ecop_scatter_base(args)

            for rltzn_lab in sim_grp_main:
                probs_a = sim_grp_main[f'{rltzn_lab}/probs'][:, di_a]

                probs_b = sim_grp_main[f'{rltzn_lab}/probs'][:, di_b]

                if ref_timing_ser.size != sim_clrs.shape[0]:
                    sim_timing_ser = np.arange(
                        1.0, probs_a.size + 1.0) / (probs_a.size + 1.0)

                    sim_clrs = plt.get_cmap(cmap_str)(sim_timing_ser)

                fig_suff = f'sim_{dl_a}_{dl_b}_{rltzn_lab}'

                args = (
                    probs_a,
                    probs_b,
                    fig_suff,
                    self._ms_dir,
                    plt_sett,
                    cmap_mappable_beta,
                    sim_clrs)

                self._plot_cross_ecop_scatter_base(args)

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site cross ecop scatters '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    @staticmethod
    def _plot_cross_ecop_scatter_base(args):

        (probs_a,
         probs_b,
         fig_suff,
         out_dir,
         plt_sett,
         cmap_mappable_beta,
         clrs) = args

        fig, axes = plt.subplots(1, 1, squeeze=False)

        row, col = 0, 0

        axes[row, col].scatter(
            probs_a,
            probs_b,
            c=clrs,
            alpha=plt_sett.alpha_1)

        axes[row, col].grid()

        axes[row, col].set_aspect('equal')

        axes[row, col].set_ylabel('Probability')
        axes[row, col].set_xlabel('Probability')

        axes[row, col].set_xlim(0, 1)
        axes[row, col].set_ylim(0, 1)

        cbaxes = fig.add_axes([0.2, 0.0, 0.65, 0.05])

        plt.colorbar(
            mappable=cmap_mappable_beta,
            cax=cbaxes,
            orientation='horizontal',
            label='Timing',
            drawedges=False)

        plt.savefig(
            str(out_dir / f'ms__cross_ecops_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return


class PhaseAnnealingPlotSingleSiteQQ:

    def _plot_qq_cmpr(self, var_label, step_lab):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ss__{var_label}_qq'

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        if step_lab is not None:
            steps = h5_hdl[f'settings/_sett_obj_{step_lab}s_vld']
            steps_opt = h5_hdl[f'settings/_sett_obj_{step_lab}s']

            loop_prod = product(data_labels, steps)

        else:
            loop_prod = combinations(data_labels, 2)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for loop_vars in loop_prod:

            if step_lab is not None:
                (data_label, step) = loop_vars

                ref_probs = h5_hdl[
                    f'data_ref_rltzn/_ref_{var_label}_qq_'
                    f'dict_{data_label}_{step:03d}'][:]

            else:
                cols = loop_vars

                ref_probs = h5_hdl[
                    f'data_ref_rltzn/_ref_{var_label}_qq_'
                    f'dict_{cols[0]}_{cols[1]}'][:]

            plt.figure()

            plt.plot(
                ref_probs,
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

                if step_lab is not None:
                    sim_probs = sim_grp_main[
                        f'{rltzn_lab}/{var_label}_'
                        f'qq_{data_label}_{step:03d}']

                else:
                    sim_probs = sim_grp_main[
                        f'{rltzn_lab}/{var_label}_qq_{cols[0]}_{cols[1]}']

                plt.plot(
                    ref_probs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Sim. F(x)')

            if step_lab is not None:
                if step in steps_opt:
                    suff = 'opt'

                else:
                    suff = 'vld'

                plt.xlabel(f'Ref. F(x) ({step_lab}(s) = {step}_{suff})')

                out_name = f'{out_name_pref}_{data_label}_{step:03d}.png'

            else:
                plt.xlabel(f'Ref. F(x)')

                out_name = f'{out_name_pref}_{"_".join(cols)}.png'

            plt.savefig(
                str(self._qq_dir / out_name), bbox_inches='tight')

            plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting single-site {var_label} QQ probabilites '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return


class PhaseAnnealingPlot(
        PhaseAnealingPlotOSV,
        PhaseAnnealingPlotSingleSite,
        PhaseAnnealingPlotMultiSite,
        PhaseAnnealingPlotSingleSiteQQ):

    def __init__(self, verbose):

        assert isinstance(verbose, bool), 'verbose not a Boolean!'

        self._vb = verbose

        self._plt_in_h5_file = None

        self._n_cpus = None

        self._plt_outputs_dir = None

        self._ss_dir = None
        self._osv_dir = None
        self._ms_dir = None

        self._qq_dir = None

        self._dens_dist_flag = False

        self._plt_input_set_flag = False
        self._plt_output_set_flag = False

        self._init_plt_settings()

        self._plt_verify_flag = False
        return

    def _init_plt_settings(self):

        '''One place to change plotting parameters for all plots'''

        fontsize = 16
        dpi = 150

        alpha_1 = 0.35
        alpha_2 = 0.6

        lw_1 = 2.0
        lw_2 = 3.0

        clr_1 = 'k'
        clr_2 = 'r'

        default_line_sett = PlotLineSettings(
            (15, 5.5),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_tols = default_line_sett
        self._plt_sett_objs = default_line_sett
        self._plt_sett_acpt_rates = default_line_sett
        self._plt_sett_phss = default_line_sett
        self._plt_sett_temps = default_line_sett
        self._plt_sett_phs_red_rates = default_line_sett
        self._plt_sett_idxs = default_line_sett
        self._plt_sett_tmrs = default_line_sett

        self._plt_sett_1D_vars = PlotLineSettings(
            (10, 10),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_1D_vars_wider = PlotLineSettings(
            (15, 10),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_ecops_denss = PlotImageSettings(
            (10, 10), dpi, fontsize, 0.9, 0.9, 'Blues')

        self._plt_sett_ecops_sctr = PlotScatterSettings(
            (10, 10), dpi, fontsize, alpha_1, alpha_2, 'C0')

        self._plt_sett_nth_ord_diffs = self._plt_sett_1D_vars

        self._plt_sett_ft_corrs = PlotLineSettings(
            (15, 10),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_mag_cdfs = self._plt_sett_1D_vars

        self._plt_sett_phs_cdfs = self._plt_sett_1D_vars

        self._plt_sett_mag_cos_sin_cdfs = self._plt_sett_1D_vars

        self._plt_sett_ts_probs = PlotLineSettings(
            (15, 7),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_phs_cross_corr_cdfs = self._plt_sett_1D_vars

        self._plt_sett_phs_cross_corr_vg = PlotLineSettings(
            (10, 10),
            dpi,
            fontsize,
            alpha_1,
            alpha_2 ,
            lw_1,
            lw_2,
            clr_1,
            clr_2)

        self._plt_sett_gnrc_cdfs = self._plt_sett_1D_vars

        self._plt_sett_cross_ecops_sctr = self._plt_sett_ecops_sctr
        self._plt_sett_cross_ft_corrs = self._plt_sett_ft_corrs
        self._plt_sett_cross_ecops_denss = self._plt_sett_ecops_denss
        self._plt_sett_cross_gnrc_cdfs = self._plt_sett_1D_vars
        self._plt_sett_cross_ecops_denss_cntmnt = self._plt_sett_ecops_denss
        return

    def set_input(
            self,
            in_h5_file,
            n_cpus,
            opt_state_vars_flag,
            single_site_flag,
            multi_site_flag,
            qq_flag):

        if self._vb:
            print_sl()

            print(
                'Setting inputs for plotting phase annealing results...\n')

        assert isinstance(in_h5_file, (str, Path))

        assert isinstance(n_cpus, (int, str)), (
            'n_cpus not an integer or a string!')

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'n_cpus can be auto only if a string!'

            n_cpus = max(1, psutil.cpu_count() - 1)

        elif isinstance(n_cpus, int):
            assert n_cpus > 0, 'Invalid n_cpus!'

        else:
            raise ValueError(n_cpus)

        in_h5_file = Path(in_h5_file)

        assert in_h5_file.exists(), 'in_h5_file does not exist!'

        assert isinstance(opt_state_vars_flag, bool), (
            'opt_state_vars_flag not a boolean!')

        assert isinstance(single_site_flag, bool), (
            'single_site_flag not a boolean!')

        assert isinstance(multi_site_flag, bool), (
            'multi_site_flag not a boolean!')

        assert isinstance(qq_flag, bool), (
            'qq_flag not a boolean!')

        assert any([
            opt_state_vars_flag,
            single_site_flag,
            multi_site_flag,
            qq_flag]), (
            'None of the plotting flags are True!')

        self._plt_in_h5_file = in_h5_file

        self._n_cpus = n_cpus

        self._plt_osv_flag = opt_state_vars_flag
        self._plt_ss_flag = single_site_flag
        self._plt_ms_flag = multi_site_flag

        self._plt_qq_flag = qq_flag

        if self._vb:
            print(
                f'Set the following input HDF5 file: '
                f'{self._plt_in_h5_file}')

            print(
                f'Optimization state variables plot flag: '
                f'{self._plt_osv_flag}')

            print(
                f'Single-site plot flag: '
                f'{self._plt_ss_flag}')

            print(
                f'Multi-site plots plot flag: '
                f'{self._plt_ms_flag}')

            print(
                f'Single-site QQ plot flag: '
                f'{self._plt_qq_flag}')

            print_el()

        self._plt_input_set_flag = True
        return

    def set_output(self, outputs_dir):

        if self._vb:
            print_sl()

            print(
                'Setting outputs directory for plotting phase annealing '
                'results...\n')

        assert isinstance(outputs_dir, (str, Path))

        outputs_dir = Path(outputs_dir)

        outputs_dir.mkdir(exist_ok=True)

        assert outputs_dir.exists(), 'Could not create outputs_dir!'

        self._plt_outputs_dir = outputs_dir

        self._ss_dir = self._plt_outputs_dir / 'single_site'

        self._osv_dir = (
            self._plt_outputs_dir / 'optimization_state_variables')

        self._ms_dir = self._plt_outputs_dir / 'multi_site'

        self._qq_dir = self._plt_outputs_dir / 'qq'

        if self._vb:
            print(
                'Set the following outputs directory:', self._plt_outputs_dir)

            print_el()

        self._plt_output_set_flag = True
        return

    def plot(self):

        if self._vb:
            print_sl()

            print('Plotting...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        ftns_args = []
        if self._plt_osv_flag:
            self._osv_dir.mkdir(exist_ok=True)

            ftns_args.extend([
                (self._plot_tols, []),
                (self._plot_obj_vals, []),
                (self._plot_acpt_rates, []),
                (self._plot_temps, []),
                (self._plot_phs_red_rates, []),
                (self._plot_idxs, []),
                (self._plot_obj_vals_indiv, []),
                (self._plot_phs_idxs_sclrs, []),
                (self._plot_tmrs, []),
                ])

        if self._plt_ss_flag:
            self._ss_dir.mkdir(exist_ok=True)

            ftns_args.extend([
                (self._plot_cmpr_1D_vars, []),
                (self._plot_cmpr_ft_corrs, []),
                (self._plot_cmpr_nth_ord_diffs, []),
                (self._plot_mag_cdfs, []),
                (self._plot_mag_cos_sin_cdfs_base, (np.cos, 'cos', 'cosine')),
                (self._plot_mag_cos_sin_cdfs_base, (np.sin, 'sin', 'sine')),
                (self._plot_ts_probs, []),
                (self._plot_phs_cdfs, []),
                (self._plot_cmpr_ecop_scatter, []),
                (self._plot_cmpr_ecop_denss, []),
                (self._plot_gnrc_cdfs_cmpr, ('scorr', 'Numerator')),
                (self._plot_gnrc_cdfs_cmpr, ('asymm_1', 'Numerator')),
                (self._plot_gnrc_cdfs_cmpr, ('asymm_2', 'Numerator')),
#                 (self._plot_gnrc_cdfs_cmpr, ('ecop_dens', 'Bin density')),
                (self._plot_gnrc_cdfs_cmpr, ('ecop_etpy', 'Bin entropy')),
                (self._plot_gnrc_cdfs_cmpr, ('pcorr', 'Numerator')),
                (self._plot_cmpr_data_ft, []),
                ])

        if self._plt_ms_flag:
            h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

            n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

            h5_hdl.close()

            if n_data_labels >= 2:

                self._ms_dir.mkdir(exist_ok=True)

                ftns_args.extend([
                    (self._plot_cross_ecop_scatter, []),
                    (self._plot_cross_ft_corrs, []),
                    (self._plot_cross_ecop_denss, []),
                    (self._plot_cross_gnrc_cdfs, ('mult_asymm_1', 'Numerator')),
                    (self._plot_cross_gnrc_cdfs, ('mult_asymm_2', 'Numerator')),
                    (self._plot_cross_ecop_denss_cntmnt, []),
                    ])

            else:
                if self._vb:
                    print('INFO: Input dataset not a multsite simulation!')

        if self._plt_qq_flag:
            self._qq_dir.mkdir(exist_ok=True)

            ftns_args.extend([
                (self._plot_qq_cmpr, ('scorr', 'lag_step')),
                (self._plot_qq_cmpr, ('asymm_1', 'lag_step')),
                (self._plot_qq_cmpr, ('asymm_2', 'lag_step')),
                (self._plot_qq_cmpr, ('ecop_etpy', 'lag_step')),
                (self._plot_qq_cmpr, ('pcorr', 'lag_step')),
                (self._plot_qq_cmpr, ('nth_ord', 'nth_ord')),
                (self._plot_qq_cmpr, ('mult_asymm_1', None)),
                (self._plot_qq_cmpr, ('mult_asymm_2', None)),
                ])

        assert ftns_args

        n_cpus = min(self._n_cpus, len(ftns_args))

        if n_cpus == 1:
            for ftn_arg in ftns_args:
                self._exec(ftn_arg)

        else:
            mp_pool = Pool(n_cpus)

            # NOTE:
            # imap_unordered does not show exceptions, map does.

#             mp_pool.imap_unordered(self._exec, ftns_args)
            mp_pool.map(self._exec, ftns_args, chunksize=1)

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

        if self._vb:
            print('Done plotting.')

            print_el()

        return

    @staticmethod
    def _exec(args):

        ftn, arg = args

        if arg:
            ftn(*arg)

        else:
            ftn()

        return

    def verify(self):

        assert self._plt_input_set_flag, 'Call set_input first!'
        assert self._plt_output_set_flag, 'Call set_output first!'

        self._plt_verify_flag = True
        return

