'''
@author: Faizan-Uni

Jan 16, 2020

1:32:31 PM
'''

import matplotlib as mpl

# has to be big enough to accomodate all plotted values
mpl.rcParams['agg.path.chunksize'] = 50000

from math import ceil
from pathlib import Path

import h5py
import numpy as np
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

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


class PhaseAnnealingPlot:

    def __init__(self, verbose):

        assert isinstance(verbose, bool), 'verbose not a Boolean!'

        self._vb = verbose

        self._plt_in_h5_file = None
        self._plt_outputs_dir = None

        self._plt_input_set_flag = False
        self._plt_output_set_flag = False

        self._init_plt_settings()

        self._plt_verify_flag = False
        return

    def _init_plt_settings(self):

        '''One place to change plotting parameters for all plots'''

        fontsize = 16
        dpi = 300

        default_line_sett = PlotLineSettings(
            (20, 7), dpi, fontsize, 0.2, 0.5, 2.0, 4.0, 'k', 'r')

        self._plt_sett_tols = default_line_sett
        self._plt_sett_objs = default_line_sett
        self._plt_sett_acpt_rates = default_line_sett
        self._plt_sett_phss = default_line_sett
        self._plt_sett_temps = default_line_sett
        self._plt_sett_phs_red_rates = default_line_sett
        self._plt_sett_idxs = default_line_sett

        self._plt_sett_1D_vars = PlotLineSettings(
            (15, 15), dpi, fontsize, 0.2, 0.7, 2.0, 4.0, 'k', 'r')

        self._plt_sett_ecops_denss = PlotImageSettings(
            (15, 15), dpi, fontsize, 0.9, 0.7, 'Blues')

        self._plt_sett_ecops_sctr = PlotScatterSettings(
            (15, 15), dpi, fontsize, 0.4, 0.7, 'C0')

        self._plt_sett_nth_ord_diffs = self._plt_sett_1D_vars

        self._plt_sett_ft_corrs = PlotLineSettings(
            (26, 15), dpi, fontsize, 0.2, 0.7, 2.0, 4.0, 'k', 'r')

        self._plt_sett_mag_cdfs = self._plt_sett_1D_vars

        self._plt_sett_phs_cdfs = self._plt_sett_1D_vars

        self._plt_sett_mag_cos_sin_cdfs = self._plt_sett_1D_vars

        self._plt_sett_ts_probs = PlotLineSettings(
            (26, 10), dpi, fontsize, 0.2, 0.7, 2.0, 4.0, 'k', 'r')
        return

    def set_input(self, in_h5_file):

        if self._vb:
            print_sl()

            print(
                'Setting input HDF5 file for plotting phase annealing '
                'results...\n')

        assert isinstance(in_h5_file, (str, Path))

        in_h5_file = Path(in_h5_file)

        assert in_h5_file.exists(), 'in_h5_file does not exist!'

        self._plt_in_h5_file = in_h5_file

        if self._vb:
            print('Set the following input HDF5 file:', self._plt_in_h5_file)

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

        if not outputs_dir.exists():
            outputs_dir.mkdir(exist_ok=True)

        assert outputs_dir.exists(), 'Could not create outputs_dir!'

        self._plt_outputs_dir = outputs_dir

        if self._vb:
            print(
                'Set the following outputs directory:', self._plt_outputs_dir)

            print_el()

        self._plt_output_set_flag = True
        return

    def plot_opt_state_vars(self):

        if self._vb:
            print_sl()

            print('Plotting optimization state variables...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        opt_state_dir = self._plt_outputs_dir / 'optimization_state_variables'

        opt_state_dir.mkdir(exist_ok=True)

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        self._plot_tols(h5_hdl, opt_state_dir)

        self._plot_obj_vals(h5_hdl, opt_state_dir)

        self._plot_acpt_rates(h5_hdl, opt_state_dir)

#         self._plot_phss(h5_hdl, opt_state_dir)

        self._plot_temps(h5_hdl, opt_state_dir)

        self._plot_phs_red_rates(h5_hdl, opt_state_dir)

#         self._plot_idxs(h5_hdl, opt_state_dir)

        h5_hdl.close()

        if self._vb:
            print('Done plotting optimization state variables.')

            print_el()
        return

    def plot_comparison(self):

        if self._vb:
            print_sl()

            print('Plotting comparision...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        cmpr_dir = self._plt_outputs_dir / 'comparison'

        cmpr_dir.mkdir(exist_ok=True)

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        self._plot_cmpr_1D_vars(h5_hdl, cmpr_dir)

        self._plot_cmpr_ft_corrs(h5_hdl, cmpr_dir)

        self._plot_cmpr_nth_ord_diffs(h5_hdl, cmpr_dir)

        self._plot_mag_cdfs(h5_hdl, cmpr_dir)

        self._plot_phs_cdfs(h5_hdl, cmpr_dir)

        self._plot_mag_cos_sin_cdfs_base(
            h5_hdl, cmpr_dir, np.cos, 'cos', 'cosine')

        self._plot_mag_cos_sin_cdfs_base(
            h5_hdl, cmpr_dir, np.sin, 'sin', 'sine')

        self._plot_ts_probs(h5_hdl, cmpr_dir)

        self._plot_cmpr_ecop_scatter(h5_hdl, cmpr_dir)

        self._plot_cmpr_ecop_denss(h5_hdl, cmpr_dir)

        h5_hdl.close()

        if self._vb:
            print('Done plotting comparision.')

            print_el()
        return

    def verify(self):

        assert self._plt_input_set_flag, 'Call set_input first!'
        assert self._plt_output_set_flag, 'Call set_output first!'

        self._plt_verify_flag = True
        return

    def _plot_ts_probs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ts_probs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_ts_probs = h5_hdl[f'data_ref_rltzn/_ref_probs']

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
                label = 'sim-ref'

            else:
                label = None

            sim_ts_probs = sim_grp_main[f'{rltzn_lab}/probs']

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

        plt.savefig(
            str(out_dir / f'cmpr__ts_probs.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phs_cdfs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        ref_phs_abs = np.sort(np.angle(h5_hdl['data_ref_rltzn/_ref_ft']))

        ref_probs = np.arange(1.0, ref_phs_abs.size + 1) / (ref_phs_abs.size + 1)

        if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:
            sim_probs = np.array([], dtype=np.float64)

        else:
            sim_probs = ref_probs

        plt.figure()

        plt.plot(
            ref_phs_abs,
            ref_probs,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        sim_grp_main = h5_hdl['data_sim_rltzns']

        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_phs_abs = np.sort(np.angle(sim_grp_main[f'{rltzn_lab}/ft']))

            if sim_probs.size != sim_phs_abs.size:
                sim_probs = np.arange(
                    1.0, sim_phs_abs.size + 1.0) / (sim_phs_abs.size + 1)

            plt.plot(
                sim_phs_abs,
                sim_probs,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        plt.grid()

        plt.legend(framealpha=0.7)

        plt.ylabel('Probability')

        plt.xlabel(f'FT Phase')

        plt.savefig(str(out_dir / f'cmpr_phs_cdfs.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_mag_cos_sin_cdfs_base(
            self, h5_hdl, out_dir, sin_cos_ftn, shrt_lab, lng_lab):

        assert sin_cos_ftn in [np.cos, np.sin], (
            'np.cos and np.sin allowed only!')

        plt_sett = self._plt_sett_mag_cos_sin_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        ref_ft = h5_hdl['data_ref_rltzn/_ref_ft']

        ref_phs = np.angle(ref_ft)

        ref_mag = np.abs(ref_ft)

        ref_mag_cos_abs = np.sort(ref_mag * sin_cos_ftn(ref_phs))

        ref_probs = np.arange(1.0, ref_mag_cos_abs.size + 1) / (
            (ref_mag_cos_abs.size + 1))

        if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:
            sim_probs = np.array([], dtype=np.float64)

        else:
            sim_probs = ref_probs

        plt.figure()

        plt.plot(
            ref_mag_cos_abs,
            ref_probs,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        sim_grp_main = h5_hdl['data_sim_rltzns']

        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_ft = sim_grp_main[f'{rltzn_lab}/ft']

            sim_phs = np.angle(sim_ft)

            sim_mag = np.abs(sim_ft)

            sim_mag_cos_abs = np.sort(sim_mag * sin_cos_ftn(sim_phs))

            if sim_probs.size != sim_mag_cos_abs.size:
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

        plt.savefig(
            str(out_dir / f'cmpr_mag_{shrt_lab}_cdfs.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_mag_cdfs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_mag_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        ref_mag_abs = np.sort(np.abs(h5_hdl['data_ref_rltzn/_ref_ft']))

        ref_probs = np.arange(1.0, ref_mag_abs.size + 1) / (ref_mag_abs.size + 1)

        if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:
            sim_probs = np.array([], dtype=np.float64)

        else:
            sim_probs = ref_probs

        plt.figure()

        plt.plot(
            ref_mag_abs,
            ref_probs,
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        sim_grp_main = h5_hdl['data_sim_rltzns']

        leg_flag = True
        for rltzn_lab in sim_grp_main:
            if leg_flag:
                label = 'sim'

            else:
                label = None

            sim_mag_abs = np.sort(np.abs(sim_grp_main[f'{rltzn_lab}/ft']))

            if sim_probs.size != sim_mag_abs.size:
                sim_probs = np.arange(
                    1.0, sim_mag_abs.size + 1.0) / (sim_mag_abs.size + 1)

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

        plt.savefig(str(out_dir / f'cmpr_mag_cdfs.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_tols(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_tols

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        beg_iters = h5_hdl['settings'].attrs['_sett_ann_obj_tol_iters']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            tol_iters = np.arange(
                sim_grp_main[f'{rltzn_lab}/tols'].shape[0]) + beg_iters

            plt.plot(
                tol_iters,
                sim_grp_main[f'{rltzn_lab}/tols'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(
            f'Mean absolute objective function\ndifference of previous '
            f'{beg_iters} iterations')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__tols.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_obj_vals(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_objs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # obj_vals_all
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

#         plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Raw objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_all.png'), bbox_inches='tight')

        plt.close()

        # obj_vals_min
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/obj_vals_min'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

#         plt.ylim(0, plt.ylim()[1])

        plt.xlabel('Iteration')

        plt.ylabel(f'Running minimum objective function value')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__obj_vals_min.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_acpt_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_acpt_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

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
            str(out_dir / f'opt_state__acpt_rates.png'), bbox_inches='tight')

        plt.close()

        # acpt_rates_dfrntl
        acpt_rate_iters = (
            h5_hdl['settings'].attrs['_sett_ann_acpt_rate_iters'])

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
            str(out_dir / f'opt_state__acpt_rates_dfrntl.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phss(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/phss_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__phss_all.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_idxs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_idxs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        # idxs_all
        plt.figure()

        for rltzn_lab in sim_grp_main:
            plt.plot(
                sim_grp_main[f'{rltzn_lab}/idxs_all'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Index')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__idxs_all.png'), bbox_inches='tight')

        plt.close()

        # idxs_acpt
        plt.figure()

        for rltzn_lab in sim_grp_main:

            idxs_acpt = sim_grp_main[f'{rltzn_lab}/idxs_acpt']

            plt.plot(
                idxs_acpt[:, 0],
                idxs_acpt[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Index')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__idxs_acpt.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_temps(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_temps

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            temps_all = sim_grp_main[f'{rltzn_lab}/temps']

            plt.plot(
                temps_all[:, 0],
                temps_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Annealing temperature')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__temps.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phs_red_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        plt.figure()

        for rltzn_lab in sim_grp_main:
            temps_all = sim_grp_main[f'{rltzn_lab}/phs_red_rates']

            plt.plot(
                temps_all[:, 0],
                temps_all[:, 1],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1)

        plt.ylim(0, 1)

        plt.xlabel('Iteration')

        plt.ylabel(f'Phase increment reduction rate')

        plt.grid()

        plt.savefig(
            str(out_dir / f'opt_state__phs_red_rates.png'),
            bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_1D_vars(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_1D_vars

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        axes = plt.subplots(2, 2, squeeze=False)[1]

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        axes[0, 0].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_scorrs'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        axes[1, 0].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_1'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        axes[1, 1].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_asymms_2'],
            alpha=plt_sett.alpha_2,
            color=plt_sett.lc_2,
            lw=plt_sett.lw_2,
            label='ref')

        axes[0, 1].plot(
            lag_steps,
            h5_hdl['data_ref_rltzn/_ref_ecop_etpy_arrs'][:],
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

            axes[0, 0].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/scorrs'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            axes[1, 0].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_1'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            axes[1, 1].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/asymms_2'],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            axes[0, 1].plot(
                lag_steps,
                sim_grp_main[f'{rltzn_lab}/ecop_entps'][:],
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        axes[0, 0].grid()
        axes[1, 0].grid()
        axes[1, 1].grid()
        axes[0, 1].grid()

        axes[0, 0].legend(framealpha=0.7)
        axes[1, 0].legend(framealpha=0.7)
        axes[1, 1].legend(framealpha=0.7)
        axes[0, 1].legend(framealpha=0.7)

        axes[0, 0].set_ylabel('Spearman correlation')

        axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

        axes[1, 1].set_xlabel('Lag steps')
        axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

        axes[0, 1].set_xlabel('Lag steps')
        axes[0, 1].set_ylabel('Entropy')

        plt.tight_layout()

        plt.savefig(
            str(out_dir / f'cmpr__scorrs_asymms_etps.png'), bbox_inches='tight')

        plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ecop_denss_base(
            self,
            lag_steps,
            fig_suff,
            vmin,
            vmax,
            ecop_denss,
            cmap_mappable_beta,
            out_dir,
            plt_sett):

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
            alpha=plt_sett.alpha_1)

        plt.savefig(
            str(out_dir / f'cmpr__ecop_denss_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_denss(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ecops_denss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        ecop_denss = h5_hdl['data_ref_rltzn/_ref_ecop_dens_arrs']

        vmin = 0.0
        vmax = np.max(ecop_denss) * 0.85

        fig_suff = 'ref'

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        cmap_mappable_beta = plt.cm.ScalarMappable(
            norm=Normalize(vmin / 100, vmax / 100, clip=True),
            cmap=cmap_beta)

        cmap_mappable_beta.set_array([])

        self._plot_cmpr_ecop_denss_base(
            lag_steps,
            fig_suff,
            vmin,
            vmax,
            ecop_denss,
            cmap_mappable_beta,
            out_dir,
            plt_sett)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for rltzn_lab in sim_grp_main:
            fig_suff = f'sim_{rltzn_lab}'
            ecop_denss = sim_grp_main[f'{rltzn_lab}/ecop_dens']

            self._plot_cmpr_ecop_denss_base(
                lag_steps,
                fig_suff,
                vmin,
                vmax,
                ecop_denss,
                cmap_mappable_beta,
                out_dir,
                plt_sett)

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ecop_scatter_base(
            self,
            lag_steps,
            fig_suff,
            probs,
            out_dir,
            plt_sett,
            cmap_mappable_beta,
            clrs):

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
            label='Timing')

        plt.savefig(
            str(out_dir / f'cmpr__ecops_scatter_{fig_suff}.png'),
            bbox_inches='tight')

        plt.close()
        return

    def _plot_cmpr_ecop_scatter(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ecops_sctr

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        probs = h5_hdl['data_ref_rltzn/_ref_probs'][:]

        fig_suff = 'ref'

        cmap_str = 'jet'

        cmap_beta = plt.get_cmap(cmap_str)

        cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap_beta)

        cmap_mappable_beta.set_array([])

        ref_timing_ser = np.arange(1.0, probs.size + 1.0) / (probs.size + 1.0)

        ref_clrs = plt.get_cmap(cmap_str)(ref_timing_ser)

        if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:

            sim_clrs = np.array([], dtype=np.float64)

        else:
            sim_timing_ser = ref_timing_ser
            sim_clrs = ref_clrs

        self._plot_cmpr_ecop_scatter_base(
            lag_steps,
            fig_suff,
            probs,
            out_dir,
            plt_sett,
            cmap_mappable_beta,
            ref_clrs)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for rltzn_lab in sim_grp_main:
            probs = sim_grp_main[f'{rltzn_lab}/probs'][:]

            if probs.size != sim_clrs.shape[0]:
                sim_timing_ser = np.arange(
                    1.0, probs.size + 1.0) / (probs.size + 1.0)

                sim_clrs = plt.get_cmap(cmap_str)(sim_timing_ser)

            fig_suff = f'sim_{rltzn_lab}'

            self._plot_cmpr_ecop_scatter_base(
                lag_steps,
                fig_suff,
                probs,
                out_dir,
                plt_sett,
                cmap_mappable_beta,
                sim_clrs)

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_nth_ord_diffs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_nth_ord_diffs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for nth_ord in nth_ords:
            ref_probs = h5_hdl[
                f'data_ref_rltzn/_ref_nth_ords_cdfs_dict_{nth_ord:03d}_y'][:]

            if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:
                sim_probs = np.array([], dtype=np.float64)

            else:
                sim_probs = ref_probs

            plt.figure()

            ref_vals = h5_hdl[
                f'data_ref_rltzn/_ref_nth_ord_diffs_{nth_ord:03d}']

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
                    f'{rltzn_lab}/sim_nth_ord_diffs_{nth_ord:03d}']

                if sim_probs.size != sim_vals.size:
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

            plt.xlabel(f'Difference (order = {nth_ord})')

            plt.savefig(
                str(out_dir / f'cmpr__nth_diff_cdfs_{nth_ord:03d}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ft_corrs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        sim_grp_main = h5_hdl['data_sim_rltzns']

        ref_cumm_corrs = h5_hdl[f'data_ref_rltzn/_ref_ft_cumm_corr']

        ref_freqs = np.arange(1, ref_cumm_corrs.size + 1)

        if h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']:
            sim_freqs = np.array([], dtype=int)

        else:
            sim_freqs = ref_freqs

        # cumm ft corrs, sim_ref
        plt.figure()

        plt.plot(
            ref_freqs,
            ref_cumm_corrs,
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

            sim_cumm_corrs = sim_grp_main[f'{rltzn_lab}/ft_cumm_corr_sim_ref']

            if sim_freqs.size != sim_cumm_corrs.size:
                sim_freqs = np.arange(1, sim_cumm_corrs.size + 1)

            plt.plot(
                sim_freqs,
                sim_cumm_corrs,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        plt.grid()

        plt.legend(framealpha=0.7)

        plt.ylabel('Cummulative correlation')

        plt.xlabel(f'Frequency')

        plt.ylim(-1, +1)

        plt.savefig(
            str(out_dir / f'cmpr__ft_cumm_corrs_sim_ref.png'),
            bbox_inches='tight')

        plt.close()

        # cumm ft corrs, sim_sim
        plt.figure()

        if ref_freqs.size != ref_cumm_corrs.size:
            ref_freqs = np.arange(1, ref_cumm_corrs.size + 1)

        plt.plot(
            ref_freqs,
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

            sim_cumm_corrs = sim_grp_main[f'{rltzn_lab}/ft_cumm_corr_sim_sim']

            if sim_freqs.size != sim_cumm_corrs.size:
                sim_freqs = np.arange(1, sim_cumm_corrs.size + 1)

            plt.plot(
                sim_freqs,
                sim_cumm_corrs,
                alpha=plt_sett.alpha_1,
                color=plt_sett.lc_1,
                lw=plt_sett.lw_1,
                label=label)

            leg_flag = False

        plt.grid()

        plt.legend(framealpha=0.7)

        plt.ylabel('Cummulative correlation')

        plt.xlabel(f'Frequency')

        plt.ylim(-1, +1)

        plt.savefig(
            str(out_dir / f'cmpr__ft_cumm_corrs_sim_sim.png'),
            bbox_inches='tight')

        plt.close()

        if ((not h5_hdl['flags'].attrs['_sett_extnd_len_set_flag']) or
            (h5_hdl['settings/_sett_extnd_len_rel_shp'][0] == 1)):

            # diff cumm ft corrs
            plt.figure()

            ref_freq_corrs = np.concatenate((
                [ref_cumm_corrs[0]],
                ref_cumm_corrs[1:] - ref_cumm_corrs[:-1]))

            plt.plot(
                ref_freqs,
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
                    f'{rltzn_lab}/ft_cumm_corr_sim_ref']

                if sim_freqs.size != sim_cumm_corrs.size:
                    sim_freqs = np.arange(1, sim_cumm_corrs.size + 1)

                sim_freq_corrs = np.concatenate((
                    [sim_cumm_corrs[0]],
                    sim_cumm_corrs[1:] - sim_cumm_corrs[:-1]))

                plt.plot(
                    sim_freqs,
                    sim_freq_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Differential correlation')

            plt.xlabel(f'Frequency')

            max_ylim = max(np.abs(plt.ylim()))

            plt.ylim(-max_ylim, +max_ylim)

            plt.savefig(
                str(out_dir / f'cmpr__ft_diff_freq_corrs_sim_ref.png'),
                bbox_inches='tight')

            plt.close()

        else:
            print('\n')

            print(
                'Did not plot differential sim_ref ft corrs due to extend '
                'flag!')

            print('\n')

        set_mpl_prms(old_mpl_prms)
        return
