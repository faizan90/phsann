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

        self._plot_phs_cross_corr_mat(h5_hdl, cmpr_dir)

#         self._plot_phs_cross_corr_vg(h5_hdl, cmpr_dir)

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

    def _get_upper_mat_corrs_with_distance(self, corrs_mat):

        n_ref_corr_vals = (
            corrs_mat.shape[0] *
            (corrs_mat.shape[0] - 1)) // 2

        upper_corrs = np.empty(n_ref_corr_vals, dtype=float)
        distances = np.empty(n_ref_corr_vals, dtype=int)

        cross_corrs_ctr = 0
        for i in range(corrs_mat.shape[0]):
            for j in range(corrs_mat.shape[1]):
                if i >= j:
                    continue

                upper_corrs[cross_corrs_ctr] = corrs_mat[i, j]

                distances[cross_corrs_ctr] = i - j

                cross_corrs_ctr += 1

        assert cross_corrs_ctr == n_ref_corr_vals

        assert np.all((upper_corrs >= -1) & (upper_corrs <= +1))

        return distances, upper_corrs

    def _plot_phs_cross_corr_vg(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_cross_corr_vg

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'cmpr_phs_cross_corr_vg'

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):
            ref_phs_cross_corr_mat = h5_hdl[
                f'data_ref_rltzn/{phs_cls_ctr}/_ref_phs_cross_corr_mat']

            ref_distances, ref_cross_corrs = (
                self._get_upper_mat_corrs_with_distance(
                ref_phs_cross_corr_mat))

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
                sim_distances = np.array([], dtype=np.float64)

            else:
                sim_distances = ref_distances

            plt.figure()

            plt.scatter(
                ref_distances,
                ref_cross_corrs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.xlabel('Frequency distance')

            plt.ylabel(f'Cross correlation')

            plt.savefig(
                str(out_dir / f'{out_name_pref}_ref_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            for rltzn_lab in sim_grp_main:

                plt.figure()

                sim_phs_cross_corr_mat = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/phs_cross_corr_mat']

                sim_distances, sim_cross_corrs = (
                    self._get_upper_mat_corrs_with_distance(
                    sim_phs_cross_corr_mat))

                if sim_distances.size != sim_cross_corrs.size:
                    sim_distances = np.arange(
                        1.0, sim_cross_corrs.size + 1.0) / (
                            (sim_cross_corrs.size + 1))

                plt.scatter(
                    sim_distances,
                    sim_cross_corrs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    label='sim')

                plt.grid()

                plt.legend(framealpha=0.7)

                plt.xlabel('Frequency distance')

                plt.ylabel(f'Cross correlation')

                out_name = f'{out_name_pref}_sim_{rltzn_lab}_{phs_cls_ctr}.png'

                plt.savefig(str(out_dir / out_name), bbox_inches='tight')

                plt.close()

        set_mpl_prms(old_mpl_prms)

        return

    def _get_upper_mat_corrs(self, corrs_mat):

        n_ref_corr_vals = (
            corrs_mat.shape[0] *
            (corrs_mat.shape[0] - 1)) // 2

        upper_corrs = np.empty(n_ref_corr_vals, dtype=float)

        cross_corrs_ctr = 0
        for i in range(corrs_mat.shape[0]):
            for j in range(corrs_mat.shape[1]):
                if i >= j:
                    continue

                upper_corrs[cross_corrs_ctr] = corrs_mat[i, j]

                cross_corrs_ctr += 1

        assert cross_corrs_ctr == n_ref_corr_vals

        assert np.all((upper_corrs >= -1) & (upper_corrs <= +1))

        return upper_corrs

    def _plot_phs_cross_corr_mat(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_cross_corr_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'cmpr_phs_cross_corr_cdfs'

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):
            ref_phs_cross_corr_mat = h5_hdl[
                f'data_ref_rltzn/{phs_cls_ctr}/_ref_phs_cross_corr_mat']

            ref_cross_corrs = self._get_upper_mat_corrs(ref_phs_cross_corr_mat)

            n_ref_corr_vals = ref_cross_corrs.size

            ref_cross_corrs.sort()

            ref_probs = np.arange(1.0, n_ref_corr_vals + 1) / (
                (n_ref_corr_vals + 1))

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
                sim_probs = np.array([], dtype=np.float64)

            else:
                sim_probs = ref_probs

            plt.figure()

            plt.plot(
                ref_cross_corrs,
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

                sim_phs_cross_corr_mat = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/phs_cross_corr_mat']

                sim_cross_corrs = self._get_upper_mat_corrs(
                    sim_phs_cross_corr_mat)

                sim_cross_corrs.sort()

                if sim_probs.size != sim_cross_corrs.size:
                    sim_probs = np.arange(
                        1.0, sim_cross_corrs.size + 1.0) / (
                            (sim_cross_corrs.size + 1))

                plt.plot(
                    sim_cross_corrs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'Cross correlation')

            plt.savefig(
                str(out_dir / f'{out_name_pref}_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_ts_probs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ts_probs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        out_name_pref = 'cmpr__ts_probs'

        sim_grp_main = h5_hdl['data_sim_rltzns']

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        for phs_cls_ctr in range(n_phs_clss):

            ref_ts_probs = h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_probs']

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
                    f'{rltzn_lab}/{phs_cls_ctr}/probs']

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
                str(out_dir / f'{out_name_pref}_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phs_cdfs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            ref_phs = np.pi + np.sort(np.angle(h5_hdl[
                f'data_ref_rltzn/{phs_cls_ctr}/_ref_ft']))

            ref_probs = np.arange(
                1.0, ref_phs.size + 1) / (ref_phs.size + 1.0)

            ref_phs_dens_y = (ref_phs[1:] - ref_phs[:-1])

            ref_phs_dens_x = ref_phs[:-1] + (0.5 * (ref_phs_dens_y))

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
                sim_probs = np.array([], dtype=np.float64)
                sim_phs_dens_x = np.array([], dtype=np.float64)

            else:
                sim_probs = ref_probs
                sim_phs_dens_x = ref_phs_dens_x

            prob_pln_fig = plt.figure()
            dens_plr_fig = plt.figure()
            dens_pln_fig = plt.figure()

            plt.figure(prob_pln_fig.number)
            plt.plot(
                ref_phs,
                ref_probs,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            plt.figure(dens_plr_fig.number)
            plt.polar(
                ref_phs_dens_x,
                ref_phs_dens_y,
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_1,
                label='ref')

            plt.figure(dens_pln_fig.number)
            plt.plot(
                ref_phs_dens_x,
                ref_phs_dens_y,
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

                sim_phs = np.pi + np.sort(
                    np.angle(sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/ft']))

                sim_phs_dens_y = (
                    (sim_phs[1:] - sim_phs[:-1]) *
                    ((sim_phs.size + 1) / (ref_phs.size + 1)))

                if sim_phs_dens_x.size != sim_phs_dens_y.size:
                    sim_probs = np.arange(
                        1.0, sim_phs.size + 1) / (sim_phs.size + 1.0)

                    sim_phs_dens_x = sim_phs[:-1] + (0.5 * (sim_phs_dens_y))

                plt.figure(prob_pln_fig.number)
                plt.plot(
                    sim_phs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                plt.figure(dens_plr_fig.number)
                plt.polar(
                    sim_phs_dens_x,
                    sim_phs_dens_y,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                plt.figure(dens_pln_fig.number)
                plt.plot(
                    sim_phs_dens_x,
                    sim_phs_dens_y,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            # probs plain
            plt.figure(prob_pln_fig.number)

            plt.grid(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Probability')

            plt.xlabel(f'FT Phase')

            plt.savefig(
                str(out_dir / f'cmpr_phs_cdfs_plain_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            # dens polar
            plt.figure(dens_plr_fig.number)

            plt.grid(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Density\n\n')

            plt.xlabel(f'FT Phase')

            plt.savefig(
                str(out_dir / f'cmpr_phs_pdfs_polar_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            # dens plain
            plt.figure(dens_pln_fig.number)

            plt.grid(True)

            plt.legend(framealpha=0.7)

            plt.ylabel('Density')

            plt.xlabel(f'FT Phase')

            plt.savefig(
                str(out_dir / f'cmpr_phs_pdfs_plain_{phs_cls_ctr}.png'),
                bbox_inches='tight')

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

        sim_grp_main = h5_hdl['data_sim_rltzns']

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        for phs_cls_ctr in range(n_phs_clss):

            ref_ft = h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_ft']

            ref_phs = np.angle(ref_ft)

            ref_mag = np.abs(ref_ft)

            ref_mag_cos_abs = np.sort(ref_mag * sin_cos_ftn(ref_phs))

            ref_probs = np.arange(1.0, ref_mag_cos_abs.size + 1) / (
                (ref_mag_cos_abs.size + 1))

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
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

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_ft = sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/ft']

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
                str(out_dir / f'cmpr_mag_{shrt_lab}_cdfs_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_mag_cdfs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_mag_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            ref_mag_abs = np.sort(np.abs(
                h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_ft']))

            ref_probs = (
                np.arange(1.0, ref_mag_abs.size + 1) / (ref_mag_abs.size + 1))

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
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

            leg_flag = True
            for rltzn_lab in sim_grp_main:
                if leg_flag:
                    label = 'sim'

                else:
                    label = None

                sim_mag_abs = np.sort(np.abs(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/ft']))

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

            plt.savefig(
                str(out_dir / f'cmpr_mag_cdfs_{phs_cls_ctr}.png'),
                bbox_inches='tight')

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

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        for phs_cls_ctr in range(n_phs_clss):
            plt.figure()

            for rltzn_lab in sim_grp_main:
                tol_iters = beg_iters + np.arange(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/tols'].shape[0])

                plt.plot(
                    tol_iters,
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/tols'],
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
                str(out_dir / f'opt_state__tols_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_obj_vals(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_objs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):
            # obj_vals_all
            plt.figure()
            for rltzn_lab in sim_grp_main:
                plt.plot(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/obj_vals_all'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.xlabel('Iteration')

            plt.ylabel(f'Raw objective function value')

            plt.grid()

            plt.savefig(
                str(out_dir / f'opt_state__obj_vals_all_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            # obj_vals_min
            plt.figure()
            for rltzn_lab in sim_grp_main:
                plt.plot(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/obj_vals_min'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.xlabel('Iteration')

            plt.ylabel(f'Running minimum objective function value')

            plt.grid()

            plt.savefig(
                str(out_dir / f'opt_state__obj_vals_min_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_acpt_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_acpt_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        acpt_rate_iters = (
            h5_hdl['settings'].attrs['_sett_ann_acpt_rate_iters'])

        for phs_cls_ctr in range(n_phs_clss):

            # acpt_rates_all
            plt.figure()
            for rltzn_lab in sim_grp_main:
                plt.plot(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/acpt_rates_all'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.ylim(0, 1)

            plt.xlabel('Iteration')

            plt.ylabel(f'Running mean acceptance rate')

            plt.grid()

            plt.savefig(
                str(out_dir / f'opt_state__acpt_rates_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            # acpt_rates_dfrntl
            plt.figure()
            for rltzn_lab in sim_grp_main:
                acpt_rate_dfrntl = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/acpt_rates_dfrntl']

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
                str(out_dir /
                    f'opt_state__acpt_rates_dfrntl_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phss(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phss

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            plt.figure()

            for rltzn_lab in sim_grp_main:
                plt.plot(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/phss_all'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.xlabel('Iteration')

            plt.ylabel(f'Phase')

            plt.grid()

            plt.savefig(
                str(out_dir / f'opt_state__phss_all_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_idxs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_idxs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            # idxs_all
            plt.figure()
            for rltzn_lab in sim_grp_main:
                plt.plot(
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/idxs_all'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1)

            plt.xlabel('Iteration')

            plt.ylabel(f'Index')

            plt.grid()

            plt.savefig(
                str(out_dir / f'opt_state__idxs_all_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

            # idxs_acpt
            plt.figure()
            for rltzn_lab in sim_grp_main:
                idxs_acpt = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/idxs_acpt']

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
                str(out_dir / f'opt_state__idxs_acpt_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_temps(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_temps

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            plt.figure()
            for rltzn_lab in sim_grp_main:
                temps_all = sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/temps']

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
                str(out_dir / f'opt_state__temps_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_phs_red_rates(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_phs_red_rates

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            plt.figure()
            for rltzn_lab in sim_grp_main:
                temps_all = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/phs_red_rates']

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
                str(out_dir / f'opt_state__phs_red_rates_{phs_cls_ctr}.png'),
                bbox_inches='tight')

            plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_1D_vars(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_1D_vars

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        lag_steps = h5_hdl['settings/_sett_obj_lag_steps']

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            axes = plt.subplots(2, 2, squeeze=False)[1]

            axes[0, 0].plot(
                lag_steps,
                h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_scorrs'],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[1, 0].plot(
                lag_steps,
                h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_asymms_1'],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[1, 1].plot(
                lag_steps,
                h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_asymms_2'],
                alpha=plt_sett.alpha_2,
                color=plt_sett.lc_2,
                lw=plt_sett.lw_2,
                label='ref')

            axes[0, 1].plot(
                lag_steps,
                h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_ecop_etpy_arrs'],
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
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/scorrs'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 0].plot(
                    lag_steps,
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/asymms_1'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[1, 1].plot(
                    lag_steps,
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/asymms_2'],
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                axes[0, 1].plot(
                    lag_steps,
                    sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/ecop_entps'],
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

            axes[1, 0].set_xlabel('Lag steps')
            axes[1, 0].set_ylabel('Asymmetry (Type - 1)')

            axes[1, 1].set_xlabel('Lag steps')
            axes[1, 1].set_ylabel('Asymmetry (Type - 2)')

            axes[0, 1].set_ylabel('Entropy')

            plt.tight_layout()

            plt.savefig(
                str(out_dir / f'cmpr__scorrs_asymms_etps_{phs_cls_ctr}.png'),
                bbox_inches='tight')

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

        cmap_beta = plt.get_cmap(plt.rcParams['image.cmap'])

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):
            fig_suff = f'ref_{phs_cls_ctr}'

            ecop_denss = h5_hdl[
                f'data_ref_rltzn/{phs_cls_ctr}/_ref_ecop_dens_arrs']

            vmin = 0.0
            vmax = np.max(ecop_denss) * 0.85

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

            for rltzn_lab in sim_grp_main:
                fig_suff = f'sim_{rltzn_lab}_{phs_cls_ctr}'

                ecop_denss = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/ecop_dens']

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

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        cmap_str = 'jet'

        cmap_beta = plt.get_cmap(cmap_str)

        for phs_cls_ctr in range(n_phs_clss):

            probs = h5_hdl[f'data_ref_rltzn/{phs_cls_ctr}/_ref_probs'][:]

            fig_suff = f'ref_{phs_cls_ctr}'

            cmap_mappable_beta = plt.cm.ScalarMappable(cmap=cmap_beta)

            cmap_mappable_beta.set_array([])

            ref_timing_ser = np.arange(
                1.0, probs.size + 1.0) / (probs.size + 1.0)

            ref_clrs = plt.get_cmap(cmap_str)(ref_timing_ser)

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
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

            for rltzn_lab in sim_grp_main:
                probs = sim_grp_main[f'{rltzn_lab}/{phs_cls_ctr}/probs'][:]

                if probs.size != sim_clrs.shape[0]:
                    sim_timing_ser = np.arange(
                        1.0, probs.size + 1.0) / (probs.size + 1.0)

                    sim_clrs = plt.get_cmap(cmap_str)(sim_timing_ser)

                fig_suff = f'sim_{rltzn_lab}_{phs_cls_ctr}'

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

        out_name_pref = 'cmpr__nth_diff_cdfs'

        nth_ords = h5_hdl['settings/_sett_obj_nth_ords']

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):
            for nth_ord in nth_ords:

                ref_probs = h5_hdl[
                    f'data_ref_rltzn/{phs_cls_ctr}/_ref_nth_ords_cdfs_'
                    f'dict_{nth_ord:03d}_y'][:]

                if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
                    sim_probs = np.array([], dtype=np.float64)

                else:
                    sim_probs = ref_probs

                ref_vals = h5_hdl[
                    f'data_ref_rltzn/{phs_cls_ctr}/_ref_nth_ord_diffs_'
                    f'{nth_ord:03d}']

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
                        f'{rltzn_lab}/{phs_cls_ctr}/sim_nth_ord_'
                        f'diffs_{nth_ord:03d}']

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

                out_name = f'{out_name_pref}_{nth_ord:03d}_{phs_cls_ctr}.png'

                plt.savefig(str(out_dir / out_name), bbox_inches='tight')

                plt.close()

        set_mpl_prms(old_mpl_prms)
        return

    def _plot_cmpr_ft_corrs(self, h5_hdl, out_dir):

        plt_sett = self._plt_sett_ft_corrs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        n_phs_clss = h5_hdl['data_sim'].attrs['_sim_phs_ann_n_clss']

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for phs_cls_ctr in range(n_phs_clss):

            ref_cumm_corrs = h5_hdl[
                f'data_ref_rltzn/{phs_cls_ctr}/_ref_ft_cumm_corr']

            ref_freqs = np.arange(1, ref_cumm_corrs.size + 1)

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
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

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/ft_cumm_corr_sim_ref']

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

            out_name = f'cmpr__ft_cumm_corrs_sim_ref_{phs_cls_ctr}.png'

            plt.savefig(str(out_dir / out_name), bbox_inches='tight')

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

                sim_cumm_corrs = sim_grp_main[
                    f'{rltzn_lab}/{phs_cls_ctr}/ft_cumm_corr_sim_sim']

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

            out_name = f'cmpr__ft_cumm_corrs_sim_sim_{phs_cls_ctr}.png'

            plt.savefig(str(out_dir / out_name), bbox_inches='tight')

            plt.close()

            if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] == 1:

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
                        f'{rltzn_lab}/{phs_cls_ctr}/ft_cumm_corr_sim_ref']

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

                out_name = (
                    f'cmpr__ft_diff_freq_corrs_sim_ref_{phs_cls_ctr}.png')

                plt.savefig(str(out_dir / out_name), bbox_inches='tight')

                plt.close()

        if h5_hdl['settings/_sett_extnd_len_rel_shp'][0] != 1:
            print('\n')

            print(
                'Did not plot differential sim_ref ft corrs due to extend '
                'flag!')

            print('\n')

        set_mpl_prms(old_mpl_prms)
        return
