'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from timeit import default_timer

import h5py
import numpy as np
import matplotlib.pyplot as plt

from .setts import get_mpl_prms, set_mpl_prms
from .base import PhaseAnnealingPlotBase as PAPB

plt.ioff()


class PhaseAnealingPlotOSV(PAPB):

    '''
    Supporting class of Plot.

    Optimization state variables' plots.
    '''

    def __init__(self, verbose):

        PAPB.__init__(self, verbose)
        return

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
            times_grp = sim_grp_main[f'{rltzn_lab}/cumm_call_durations']

            n_calls_grp = sim_grp_main[f'{rltzn_lab}/cumm_n_calls']

            # Times
            plt.figure(call_times_fig.number)
            for i, lab in enumerate(times_grp.attrs):
                plt.scatter(
                    i,
                    times_grp.attrs[lab],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1)

            # No. calls
            plt.figure(n_calls_fig.number)
            for i, lab in enumerate(n_calls_grp.attrs):
                plt.scatter(
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

        plt.gca().set_axisbelow(True)

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

        plt.gca().set_axisbelow(True)

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

        plt.gca().set_axisbelow(True)

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

        beg_iters = h5_hdl['settings'].attrs['sett_ann_obj_tol_iters']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            tol_iters = beg_iters + np.arange(
                sim_grp_main[f'{rltzn_lab}/tols'].shape[0])

            plt.semilogy(
                tol_iters,
                sim_grp_main[f'{rltzn_lab}/tols'][:],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean absolute objective function\ndifference of previous '
            f'{beg_iters} iterations')

        plt.grid()

        plt.gca().set_axisbelow(True)

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

        obj_flag_vals = h5_hdl['settings/sett_obj_flag_vals'][...]

        obj_flag_labels = h5_hdl['settings/sett_obj_flag_labels'][...]
        obj_flag_labels = [
            obj_flag_label.decode('utf-8')
            for obj_flag_label in obj_flag_labels]

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

            plt.gca().set_axisbelow(True)

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
                sim_grp_main[f'{rltzn_lab}/obj_vals_all'][:],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Raw objective function value')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._osv_dir / f'osv__obj_vals_all.png'),
            bbox_inches='tight')

        plt.close()

        # obj_vals_min
        plt.figure()
        for rltzn_lab in sim_grp_main:
            plt.semilogy(
                sim_grp_main[f'{rltzn_lab}/obj_vals_min'][:],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Running minimum objective function value')

        plt.grid()

        plt.gca().set_axisbelow(True)

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
            h5_hdl['settings'].attrs['sett_ann_acpt_rate_iters'])

        # acpt_rates_all
        plt.figure()
        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/acpt_rates_all'][:],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Running mean acceptance rate')

        plt.grid()

        plt.gca().set_axisbelow(True)

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

        plt.gca().set_axisbelow(True)

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

        plot_ctr = 0
        for rltzn_lab in sim_grp_main:
            idxs_all = sim_grp_main[f'{rltzn_lab}/n_idxs_all_cts'][...]

            idxs_acpt = sim_grp_main[f'{rltzn_lab}/n_idxs_acpt_cts'][...]

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

            plot_ctr += 1

            if plot_ctr == self._plt_max_n_sim_plots:
                break

        # idxs_all
        plt.figure(idxs_all_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Raw frequency')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._osv_dir / f'osv__idxs_all_hist.png'),
            bbox_inches='tight')

        plt.close()

        # idxs_acpt
        plt.figure(idxs_acpt_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Acceptance frequency')

        plt.grid()

        plt.gca().set_axisbelow(True)

        plt.savefig(
            str(self._osv_dir / f'osv__idxs_acpt_hist.png'),
            bbox_inches='tight')

        plt.close()

        # idxs_acpt_rel
        plt.figure(idxs_acpt_rel_hist_fig.number)
        plt.xlabel('Index')
        plt.ylabel(f'Relative acceptance frequency')

        plt.grid()

        plt.gca().set_axisbelow(True)

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

        plt.gca().set_axisbelow(True)

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

        plt.gca().set_axisbelow(True)

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

