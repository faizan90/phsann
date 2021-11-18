'''
Created on Nov 18, 2021

@author: Faizan3800X-Uni
'''

from .setts import PlotLineSettings, PlotImageSettings, PlotScatterSettings


class PhaseAnnealingPlotBase:

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
