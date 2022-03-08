'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''
from timeit import default_timer
from multiprocessing import Manager, Lock
from pathos.multiprocessing import ProcessPool

import numpy as np
from scipy.stats import rankdata

from gnrctsgenr import (
    GTGBase,
    GTGAlgLagNthWts,
    GTGAlgLabelWts,
    GTGAlgAutoObjWts,
    )


class PhaseAnnealingAlgLagNthWts(GTGAlgLagNthWts):

    def __init__(self):

        GTGAlgLagNthWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_lag_nth_wts(self, phs_red_rate, idxs_sclr):

        self._init_lag_nth_wts()

        self._alg_wts_lag_nth_search_flag = True

        for _ in range(self._sett_wts_lags_nths_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_lag_nth_search_flag = False

        self._update_lag_nth_wts()
        return


class PhaseAnnealingAlgLabelWts(GTGAlgLabelWts):

    def __init__(self):

        GTGAlgLabelWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_label_wts(self, phs_red_rate, idxs_sclr):

        self._init_label_wts()

        self._alg_wts_label_search_flag = True

        for _ in range(self._sett_wts_label_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_label_search_flag = False

        self._update_label_wts()
        return


class PhaseAnnealingAlgAutoObjWts(GTGAlgAutoObjWts):

    def __init__(self):

        GTGAlgAutoObjWts.__init__(self)
        return

    @GTGBase._timer_wrap
    def _set_auto_obj_wts(self, phs_red_rate, idxs_sclr):

        self._sett_wts_obj_wts = None
        self._alg_wts_obj_raw = []
        self._alg_wts_obj_search_flag = True

        for _ in range(self._sett_wts_obj_n_iters):
            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(phs_red_rate, idxs_sclr)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            self._get_obj_ftn_val()

        self._alg_wts_obj_raw = np.array(
            self._alg_wts_obj_raw, dtype=np.float64)

        assert self._alg_wts_obj_raw.ndim == 2
        assert self._alg_wts_obj_raw.shape[0] > 1

        self._update_obj_wts()

        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False
        return


class PhaseAnnealingAlgLimPtrb:

    def __init__(self):

        self._alg_lim_phsrand_ptrb_ratios = None
        self._alg_lim_phsrand_ptrb_obj_vals = None
        self._alg_lim_phsrand_ptrb_obj_val = None
        self._alg_lim_phsrand_ptrb_ratio = None
        self._alg_lim_phsrand_sel_stat = 'mean'
        return

    def _set_lim_phsrand_ptrb_ratio(self):

        self._alg_lim_phsrand_ptrb_obj_val = 0.5 * (
            self._sett_lim_phsrand_obj_lbd + self._sett_lim_phsrand_obj_ubd)

        stat_obj_vals = getattr(np, self._alg_lim_phsrand_sel_stat)(
            self._alg_lim_phsrand_ptrb_obj_vals, axis=1)

        assert np.all(np.isfinite(stat_obj_vals)), (
            'Invalid values in stat_obj_vals!')

        assert np.all(stat_obj_vals >= 0), (
            'Values zero or less in the stat_obj_vals!')

        assert np.any(stat_obj_vals <= self._sett_lim_phsrand_obj_lbd), (
            f'No values smaller than the lower perturbation objective '
            f'function value!\n{stat_obj_vals}')

        assert np.any(stat_obj_vals >= self._sett_lim_phsrand_obj_ubd), (
            f'No values larger than the upper perturbation objective '
            f'function value!\n{stat_obj_vals}')

        srt_idxs = np.argsort(stat_obj_vals)

        self._alg_lim_phsrand_ptrb_ratio = np.interp(
            self._alg_lim_phsrand_ptrb_obj_val,
            stat_obj_vals[srt_idxs],
            self._alg_lim_phsrand_ptrb_ratios[srt_idxs],
            left=-np.inf,
            right=+np.inf)

        assert (
            self._alg_lim_phsrand_ptrb_ratios.min() <=
            self._alg_lim_phsrand_ptrb_ratio <=
            self._alg_lim_phsrand_ptrb_ratios.max()), (
                f'Final perturbation ratio '
                f'({self._alg_lim_phsrand_ptrb_ratio:1.3E}) '
                f'out of subsetted perturbation bounds!')

        return

    def _plot_lim_phsrand_obj_vals(self):

        perturb_minima = self._alg_lim_phsrand_ptrb_obj_vals.min(axis=1)
        perturb_means = self._alg_lim_phsrand_ptrb_obj_vals.mean(axis=1)
        perturb_maxima = self._alg_lim_phsrand_ptrb_obj_vals.max(axis=1)
        #======================================================================

        import matplotlib.pyplot as plt
        from adjustText import adjust_text

        plt.figure(figsize=(10, 7))

        for i in range(self._alg_lim_phsrand_ptrb_obj_vals.shape[0]):
            vals = np.sort(self._alg_lim_phsrand_ptrb_obj_vals[i,:])
            probs = rankdata(vals, method='max') / (vals.size + 1.0)

            plt.plot(
                vals,
                probs,
                alpha=0.5,
                c='k',
                zorder=1)

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Obj. val.')
        plt.ylabel('Non-exceedence probability')

        out_fig_path = (
            self._sett_lim_phsrand_dir / f'perturb_obj_vals_stat_all.png')

        plt.savefig(str(out_fig_path), bbox_inches='tight')

        plt.close()
        #======================================================================

        plt.figure(figsize=(10, 7))

        vals = np.sort(perturb_minima)
        probs = rankdata(vals, method='max') / (vals.size + 1.0)

        plt.semilogx(
            vals,
            probs,
            alpha=0.75,
            c='C0',
            lw=2,
            zorder=1,
            label='minima')

        vals = np.sort(perturb_means)
        probs = rankdata(vals, method='max') / (vals.size + 1.0)

        plt.semilogx(
            vals,
            probs,
            alpha=0.75,
            c='C1',
            lw=2,
            zorder=1,
            label='mean')

        vals = np.sort(perturb_maxima)
        probs = rankdata(vals, method='max') / (vals.size + 1.0)

        plt.semilogx(
            vals,
            probs,
            alpha=0.75,
            c='C2',
            lw=2,
            zorder=1,
            label='maxima')

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Obj. val.')
        plt.ylabel('Non-exceedence probability')

        plt.legend()

        out_fig_path = (
            self._sett_lim_phsrand_dir / f'perturb_obj_vals_stats_cdfs.png')

        plt.savefig(str(out_fig_path), bbox_inches='tight')

        plt.close()
        #======================================================================

        plt.figure(figsize=(10, 7))

        plt.semilogy(
            self._alg_lim_phsrand_ptrb_ratios,
            perturb_minima,
            alpha=0.75,
            c='C0',
            lw=2,
            zorder=1,
            label='minima')

        plt.semilogy(
            self._alg_lim_phsrand_ptrb_ratios,
            perturb_means,
            alpha=0.75,
            c='C1',
            lw=2,
            zorder=1,
            label='mean')

        plt.semilogy(
            self._alg_lim_phsrand_ptrb_ratios,
            perturb_maxima,
            alpha=0.75,
            c='C2',
            lw=2,
            zorder=1,
            label='maxima')

        plt.vlines(
            self._alg_lim_phsrand_ptrb_ratio,
            0,
            self._alg_lim_phsrand_ptrb_obj_val,
            alpha=0.5,
            ls='--',
            lw=1.5,
            color='k',
            zorder=3)

        plt.hlines(
            self._alg_lim_phsrand_ptrb_obj_val,
            0,
            self._alg_lim_phsrand_ptrb_ratio,
            alpha=0.5,
            ls='--',
            lw=1.5,
            color='k',
            zorder=3)

        plt.scatter(
            [self._alg_lim_phsrand_ptrb_ratio],
            [self._alg_lim_phsrand_ptrb_obj_val],
            alpha=0.75,
            c='k',
            label='selected',
            zorder=4)

        ptexts = []
        ptext = plt.text(
            self._alg_lim_phsrand_ptrb_ratio,
            self._alg_lim_phsrand_ptrb_obj_val,
            f'({self._alg_lim_phsrand_ptrb_ratio:1.3E}, '
            f'{self._alg_lim_phsrand_ptrb_obj_val:1.3E})',
            color='k',
            alpha=0.90,
            zorder=5)

        ptexts.append(ptext)

        adjust_text(ptexts, only_move={'points': 'y', 'text': 'y'})

        plt.grid()
        plt.gca().set_axisbelow(True)

        plt.xlabel('Perturbation ratio')
        plt.ylabel('Obj. val.')

        plt.legend()

        out_fig_path = (
            self._sett_lim_phsrand_dir / f'perturb_obj_vals_stats_raw.png')

        plt.savefig(str(out_fig_path), bbox_inches='tight')

        plt.close()
        return

    def _cmpt_lim_phsrand_obj_vals_single(self, args):

        i, ptrb_ratio, = args

        perturb_obj_vals = np.empty(self._sett_lim_phsrand_iters_per_atpt)

        for j in range(self._sett_lim_phsrand_iters_per_atpt):

            # Spectra are reset to the observed.
            self._rs.ft = self._rr.ft.copy()
            self._rs.phs_spec = self._rr.phs_spec.copy()
            self._rs.mag_spec = self._rr.mag_spec.copy()

            (_,
             new_phss,
             _,
             new_coeffs,
             new_idxs) = self._get_next_iter_vars(ptrb_ratio, 1.0)

            self._update_sim(new_idxs, new_phss, new_coeffs, False)

            perturb_obj_vals[j] = self._get_obj_ftn_val().mean()

        if False:
            print(
                f'{i:04d}',
                f'{self._sett_lim_phsrand_n_ptrb_vals:04d}',
                f'{ptrb_ratio:6.3E}',
                f'{perturb_obj_vals.min():10.3f}',
                f'{perturb_obj_vals.mean():10.3f}',
                f'{perturb_obj_vals.max():10.3f}')

        return (i, perturb_obj_vals)

    @GTGBase._timer_wrap
    def _cmpt_lim_phsrand_obj_vals(self, phs_red_rate, idxs_sclr):

        beg_tm = default_timer()

        _ = phs_red_rate
        _ = idxs_sclr

        self._sett_lim_phsrand_dir.mkdir(exist_ok=True)

        ptrb_ratios = np.linspace(
            self._sett_lim_phsrand_ptrb_lbd,
            self._sett_lim_phsrand_ptrb_ubd,
            self._sett_lim_phsrand_n_ptrb_vals,
            endpoint=True)

        n_ptrb_ratios = ptrb_ratios.size

        perturb_obj_vals = np.empty(
            (n_ptrb_ratios, self._sett_lim_phsrand_iters_per_atpt))

        n_cpus = min(n_ptrb_ratios, self._sett_misc_n_cpus)

        search_attempts = 0

        ress = []

        sel_stat_ftn = getattr(np, self._alg_lim_phsrand_sel_stat)

        ubd_sclr = 1.2

        if n_cpus > 1:
            self._lock = Manager().Lock()

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            for i in range(0, n_ptrb_ratios, n_cpus):

                end_idx = min(n_ptrb_ratios, n_cpus + i)

                assert i < end_idx, 'This was not supposed to happen!'

                search_attempts += end_idx - i

                # Don't use ret_mp_idxs, it will be inefficient.
                args_gen = ((j, ptrb_ratios[j]) for j in range(i, end_idx))

                perturb_obj_vals_iter = (
                    list(mp_pool.imap(
                        self._cmpt_lim_phsrand_obj_vals_single, args_gen)))

                ress.extend(perturb_obj_vals_iter)

                if np.any(
                    [sel_stat_ftn(perturb_obj_vals_iter[k][1]) >=
                     (self._sett_lim_phsrand_obj_ubd * ubd_sclr)
                     for k in range(len(perturb_obj_vals_iter))]):

                    break

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

        else:
            self._lock = Lock()

            for i in range(n_ptrb_ratios):
                search_attempts += 1

                ress.append(self._cmpt_lim_phsrand_obj_vals_single(
                    (i, ptrb_ratios[i])))

                if (sel_stat_ftn(ress[-1][1]) >=
                    (self._sett_lim_phsrand_obj_ubd * ubd_sclr)):

                    break

            self._lock = None

        # TODO: Add a check
        # for bounds of interpolated ratios and obj vals.

        take_idxs = []
        for res in ress:
            take_idxs.append(res[0])
            perturb_obj_vals[take_idxs[-1],:] = res[1]

        take_idxs = np.array(take_idxs)

        ptrb_ratios = ptrb_ratios[take_idxs]
        perturb_obj_vals = perturb_obj_vals[take_idxs]

        res = ress = None

        assert np.all(np.isfinite(ptrb_ratios)), (
            'Invalid values in perturb_ratios!')

        assert np.all(ptrb_ratios >= 0), (
            'Values less than zero in ptrb_ratios!')

        assert np.all(np.isfinite(perturb_obj_vals)), (
            'Invalid values in perturb_obj_vals!')

        assert np.all(perturb_obj_vals >= 0), (
            'Values less than zero in perturb_obj_vals!')

        self._alg_lim_phsrand_ptrb_ratios = ptrb_ratios
        self._alg_lim_phsrand_ptrb_obj_vals = perturb_obj_vals

        self._set_lim_phsrand_ptrb_ratio()

        self._plot_lim_phsrand_obj_vals()

        end_tm = default_timer()

        if self._vb:
            print(
                f'Found perturbation ratio of '
                f'{self._alg_lim_phsrand_ptrb_ratio:5.3E} in '
                f'{end_tm - beg_tm:0.1f} '
                f'seconds using {search_attempts} attempts.')

        return
