'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

import sys
import traceback as tb
from pathlib import Path
from multiprocessing import Pool

import h5py
import numpy as np
import matplotlib.pyplot as plt

from ...misc import print_sl, print_el, get_n_cpus
from .obj import PhaseAnealingPlotOSV
from .ss import PhaseAnnealingPlotSingleSite
from .ms import PhaseAnnealingPlotMultiSite
from .qq import PhaseAnnealingPlotSingleSiteQQ
from .setts import PlotLineSettings, PlotImageSettings, PlotScatterSettings

plt.ioff()


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

        self._plt_max_n_sim_plots = None

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
            qq_flag,
            max_sims_to_plot):

        if self._vb:
            print_sl()

            print(
                'Setting inputs for plotting phase annealing results...\n')

        assert isinstance(in_h5_file, (str, Path))

        assert isinstance(n_cpus, (int, str)), (
            'n_cpus not an integer or a string!')

        if isinstance(n_cpus, str):
            assert n_cpus == 'auto', 'n_cpus can be auto only if a string!'

            n_cpus = get_n_cpus()

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

        assert isinstance(max_sims_to_plot, int), (
            'max_sims_to_plot not an integer!')

        assert max_sims_to_plot > 0, (
            'max_sims_to_plot should be greater than zero!')

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

        self._plt_max_n_sim_plots = max_sims_to_plot

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

            print(
                f'Maximum number of simulations to plot: '
                f'{self._plt_max_n_sim_plots}')

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
#                 (self._plot_cmpr_ecop_denss, []),
                (self._plot_gnrc_cdfs_cmpr, ('scorr', 'Numerator')),
                (self._plot_gnrc_cdfs_cmpr, ('asymm_1', 'Numerator')),
                (self._plot_gnrc_cdfs_cmpr, ('asymm_2', 'Numerator')),
                (self._plot_gnrc_cdfs_cmpr, ('ecop_dens', 'Bin density')),
#                 (self._plot_gnrc_cdfs_cmpr, ('ecop_etpy', 'Bin entropy')),
                (self._plot_gnrc_cdfs_cmpr, ('pcorr', 'Numerator')),
                (self._plot_cmpr_data_ft, []),
                (self._plot_cmpr_probs_ft, []),
                (self._plot_cmpr_diffs_ft_lags, ('asymm_1',)),
                (self._plot_cmpr_diffs_ft_lags, ('asymm_2',)),
                (self._plot_cmpr_diffs_ft_nth_ords, ('nth_ord',)),
                (self._plot_cmpr_etpy_ft, [])
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
                    (self._plot_cross_gnrc_cdfs, ('mult_asymm_1_diffs', 'Numerator')),
                    (self._plot_cross_gnrc_cdfs, ('mult_asymm_2_diffs', 'Numerator')),
                    (self._plot_cross_gnrc_cdfs, ('mult_ecop_dens', 'Numerator')),
                    (self._plot_cross_ecop_denss_cntmnt, []),
                    (self._plot_cmpr_cross_cmpos_ft, ('asymm_1',)),
                    (self._plot_cmpr_cross_cmpos_ft, ('asymm_2',)),
                    (self._plot_cmpr_cross_cmpos_ft, ('etpy',)),
                    ])

            else:
                self._plt_ms_flag = False

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
                ])

            if self._plt_ms_flag:
                ftns_args.extend([
                    (self._plot_qq_cmpr, ('mult_asymm_1', None)),
                    (self._plot_qq_cmpr, ('mult_asymm_2', None)),
                    (self._plot_qq_cmpr, ('mult_ecop_dens', None)),
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

        try:
            if arg:
                ftn(*arg)

            else:
                ftn()

        except:
            pre_stack = tb.format_stack()[:-1]

            err_tb = list(tb.TracebackException(*sys.exc_info()).format())

            lines = [err_tb[0]] + pre_stack + err_tb[2:]

            for line in lines:
                print(line, file=sys.stderr, end='')

        return

    def verify(self):

        assert self._plt_input_set_flag, 'Call set_input first!'
        assert self._plt_output_set_flag, 'Call set_output first!'

        self._plt_verify_flag = True
        return
