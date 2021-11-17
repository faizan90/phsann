'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''
import matplotlib as mpl
# Has to be big enough to accomodate all plotted values.
mpl.rcParams['agg.path.chunksize'] = 50000

from timeit import default_timer
from itertools import product, combinations

import h5py
import matplotlib.pyplot as plt

from .setts import get_mpl_prms, set_mpl_prms

plt.ioff()


class PhaseAnnealingPlotSingleSiteQQ:

    '''
    Supporting class of Plot. Doesn't have __init__ of it own.

    QQ-transform plots.
    '''

    def _plot_qq_cmpr(self, var_label, step_lab):

        beg_tm = default_timer()

        h5_hdl = h5py.File(self._plt_in_h5_file, mode='r', driver=None)

        plt_sett = self._plt_sett_gnrc_cdfs

        new_mpl_prms = plt_sett.prms_dict

        old_mpl_prms = get_mpl_prms(new_mpl_prms.keys())

        set_mpl_prms(new_mpl_prms)

        data_labels = tuple(h5_hdl['data_ref'].attrs['data_ref_labels'])

        if step_lab is not None:
            # For the single-site case.
            steps = h5_hdl[f'settings/sett_obj_{step_lab}s_vld'][:]
            steps_opt = h5_hdl[f'settings/sett_obj_{step_lab}s'][:]

            loop_prod = product(data_labels, steps)

            out_name_pref = f'ss__{var_label}_qq'

        else:
            # For the multi-site case.
            loop_prod = combinations(data_labels, 2)

            out_name_pref = f'ms__{var_label}_qq'

        sim_grp_main = h5_hdl['data_sim_rltzns']

        for loop_vars in loop_prod:

            if step_lab is not None:
                (data_label, step) = loop_vars

                ref_probs = h5_hdl[
                    f'data_ref_rltzn/{var_label}_qq_'
                    f'dict_{data_label}_{step:03d}'][:]

            else:
                cols = loop_vars

                ref_probs = h5_hdl[
                    f'data_ref_rltzn/{var_label}_qq_'
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
                        f'qq_dict_{data_label}_{step:03d}'][:]

                else:
                    sim_probs = sim_grp_main[
                        f'{rltzn_lab}/{var_label}_qq_dict_'
                        f'{cols[0]}_{cols[1]}'][:]

                plt.plot(
                    ref_probs,
                    sim_probs,
                    alpha=plt_sett.alpha_1,
                    color=plt_sett.lc_1,
                    lw=plt_sett.lw_1,
                    label=label)

                leg_flag = False

            plt.grid()

            plt.gca().set_axisbelow(True)

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
