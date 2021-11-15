'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

import numpy as np

from ...misc import sci_round
from ..prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgAutoObjWts:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _update_obj_wts(self, raw_wts):

        '''
        Less weights assigned to objective values that are bigger, relatively.

        Based on:
        (wts * means).sum() == means.sum()
        '''

        # Max seems to perform better than min and mean.
        means = np.array(raw_wts).max(axis=0)

        assert np.all(np.isfinite(means))

        sum_means = means.sum()

        wts = []
        for i in range(means.size):
            wt = sum_means / means[i]
            wts.append(wt)

        wts = np.array(wts)

        assert np.all(np.isfinite(wts))

        wts = (wts.size * wts) / wts.sum()

        wts_sclr = sum_means / (means * wts).sum()

        wts *= wts_sclr

        assert np.isclose((wts * means).sum(), means.sum())

        wts = sci_round(wts)

        self._sett_wts_obj_wts = wts
        return

    @PAP._timer_wrap
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

        self._update_obj_wts(self._alg_wts_obj_raw)

        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False
        return

