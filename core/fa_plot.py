'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from timeit import default_timer
from multiprocessing import Pool

import h5py
import matplotlib.pyplot as plt;plt.ioff()

from gnrctsgenr import (
    get_mpl_prms,
    set_mpl_prms,
    GTGPlotBase,
    GTGPlotOSV,
    GTGPlotSingleSite,
    GTGPlotMultiSite,
    GTGPlotSingleSiteQQ,
    GenericTimeSeriesGeneratorPlot,
    )

from gnrctsgenr.misc import print_sl, print_el


class PhaseAnnealingPlot(
        GTGPlotBase,
        GTGPlotOSV,
        GTGPlotSingleSite,
        GTGPlotMultiSite,
        GTGPlotSingleSiteQQ,
        GenericTimeSeriesGeneratorPlot):

    def __init__(self, verbose):

        GTGPlotBase.__init__(self, verbose)
        GTGPlotOSV.__init__(self)
        GTGPlotSingleSite.__init__(self)
        GTGPlotMultiSite.__init__(self)
        GTGPlotSingleSiteQQ.__init__(self)
        GenericTimeSeriesGeneratorPlot.__init__(self)
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

    def plot(self):

        if self._vb:
            print_sl()

            print('Plotting...')

        assert self._plt_verify_flag, 'Plot in an unverified state!'

        ftns_args = []

        self._fill_osv_args_gnrc(ftns_args)

        # Variables specific to PA.
        if self._plt_osv_flag:
            ftns_args.extend([
                (self._plot_phs_red_rates, []),
                (self._plot_phs_idxs_sclrs, []),
                ])

        self._fill_ss_args_gnrc(ftns_args)

        self._fill_ms_args_gnrc(ftns_args)

        self._fill_qq_args_gnrc(ftns_args)

        assert ftns_args

        n_cpus = min(self._n_cpus, len(ftns_args))

        if n_cpus == 1:
            for ftn_arg in ftns_args:
                self._exec(ftn_arg)

        else:
            mp_pool = Pool(n_cpus)

            # NOTE:
            # imap_unordered does not show exceptions, map does.

            # mp_pool.imap_unordered(self._exec, ftns_args)

            mp_pool.map(self._exec, ftns_args, chunksize=1)

            mp_pool.close()
            mp_pool.join()

            mp_pool = None

        if self._vb:
            print('Done plotting.')

            print_el()

        return
