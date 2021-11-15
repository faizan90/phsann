'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from math import factorial
from timeit import default_timer
from itertools import combinations

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from matplotlib.colors import Normalize

from fcopulas import fill_bi_var_cop_dens

from .setts import get_mpl_prms, set_mpl_prms

plt.ioff()


class PhaseAnnealingPlotMultiSite:

    '''
    Supporting class of Plot. Doesn't have __init__ of it own.

    Multi-site plots.
    '''

    def _plot_cmpr_cross_cmpos_ft(self, var_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ms__{var_label}_cmpos_ft_cumsum'

        comb_size = 2
        n_data_labels = h5_hdl['data_ref'].attrs['_data_ref_n_labels']

        n_combs = int(
            factorial(n_data_labels) /
            (factorial(comb_size) *
             factorial(n_data_labels - comb_size)))

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_vals = h5_hdl[
            f'data_ref_rltzn/_ref_mult_{var_label}_cmpos_ft_dict'][:]

        ref_periods = ((ref_vals.size - n_combs) * 2) / (
            np.arange(1, ref_vals.size - n_combs + 1))

        add_ref_periods = []
        for i in range(n_combs):
            add_ref_periods.append(ref_periods[0] * (2 + i))

        ref_periods = np.concatenate((add_ref_periods[::-1], ref_periods))

        assert ref_vals.size == ref_periods.size

        # cumm ft corrs, sim_ref
        plt.figure()

        plt.semilogx(
            ref_periods,
            ref_vals,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        plt.axvline(
            ref_periods[n_combs],
            0,
            1,
            color='b',
            alpha=plt_sett.alpha_2,
            lw=plt_sett.lw_2)

        sim_periods = None
        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_vals = sim_grp_main[
                f'{rltzn_lab}/mult_{var_label}_cmpos_ft'][:]

            if sim_periods is None:
                sim_periods = ((sim_vals.size - n_combs) * 2) / (
                    np.arange(1, sim_vals.size - n_combs + 1))

                add_sim_periods = []
                for i in range(n_combs):
                    add_sim_periods.append(sim_periods[0] * (2 + i))

                sim_periods = np.concatenate(
                    (add_sim_periods[::-1], sim_periods))

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

        plt.ylabel('Cummulative cmpos FT correlation')
        plt.xlabel(f'Period (steps)')

        out_name = f'{out_name_pref}.png'

        plt.savefig(str(self._ms_dir / out_name), bbox_inches='tight')

        plt.close()

        h5_hdl.close()

        set_mpl_prms(old_mpl_prms)

        end_tm = default_timer()

        if self._vb:
            print(
                f'Plotting multi-site {var_label} cmpos FT '
                f'took {end_tm - beg_tm:0.2f} seconds.')
        return

    def _plot_cross_gnrc_cdfs(self, var_label, x_ax_label):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = f'ms__cross_{var_label}_cdfs'

        data_labels = tuple(h5_hdl['data_ref'].attrs['_data_ref_labels'])

        combs = combinations(data_labels, 2)

        loop_prod = combs

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for cols in loop_prod:

            assert len(cols) == 2

            ref_probs = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_y'][:]

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_{var_label}_cdfs_'
                f'dict_{cols[0]}_{cols[1]}_x'][:]

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
                    f'{rltzn_lab}/{var_label}_{cols[0]}_{cols[1]}'][:]

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

        cmap_beta.colors[1,:] = [1, 1, 1, 1]

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

            cntmnt_ecop_dens_arr[
                ref_ecop_dens_arr < sim_ecop_dens_mins_arr] = -1

            cntmnt_ecop_dens_arr[
                ref_ecop_dens_arr > sim_ecop_dens_maxs_arr] = +1

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
                cmap=cmap_beta,
                shading='auto')

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
                str(self._ms_dir /
                    f'ms__cross_ecops_denss_cmpr_{fig_suff}.png'),
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

            plot_ctr = 0
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

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

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
            alpha=plt_sett.alpha_1,
            shading='auto')

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
                    sim_periods = (sim_ft_cumm_corr.size * 2) / (
                        np.arange(1, sim_ft_cumm_corr.size + 1))

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

            plot_ctr = 0
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

                plot_ctr += 1

                if plot_ctr == self._plt_max_n_sim_plots:
                    break

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
