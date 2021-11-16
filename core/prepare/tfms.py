'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from math import factorial
from itertools import combinations

import numpy as np
from scipy.stats import rankdata

from fcopulas import (
    asymms_exp,
    get_asymm_1_max,
    get_asymm_2_max,
    get_etpy_min,
    get_etpy_max,
    )

from ...misc import (
    get_local_entropy_ts_cy,
    )


class PhaseAnnealingPrepareTfms:

    '''
    Supporting class of Prepare.

    Has no verify method or any private variables of its own.
    '''

    def _get_probs(self, data, make_like_ref_flag=False):

        probs_all = np.empty_like(data, dtype=np.float64)

        for i in range(self._data_ref_n_labels):
            probs = rankdata(data[:, i]) / (data.shape[0] + 1.0)

            if make_like_ref_flag:
                assert self._rr.probs_srtd is not None

                probs = self._rr.probs_srtd[
                    np.argsort(np.argsort(probs)), i]

            probs_all[:, i] = probs

        return probs_all

    def _get_asymm_1_max(self, scorr):

        return get_asymm_1_max(scorr)

    def _get_asymm_2_max(self, scorr):

        return get_asymm_2_max(scorr)

    def _get_etpy_min(self, n_bins):

        return get_etpy_min(n_bins)

    def _get_etpy_max(self, n_bins):

        return get_etpy_max(n_bins)

    def _get_cumm_ft_corr(self, ref_ft, sim_ft):

        ref_mag = np.abs(ref_ft)
        ref_phs = np.angle(ref_ft)

        sim_mag = np.abs(sim_ft)
        sim_phs = np.angle(sim_ft)

        numr = (
            ref_mag[1:-1,:] *
            sim_mag[1:-1,:] *
            np.cos(ref_phs[1:-1,:] - sim_phs[1:-1,:]))

        demr = (
            ((ref_mag[1:-1,:] ** 2).sum(axis=0) ** 0.5) *
            ((sim_mag[1:-1,:] ** 2).sum(axis=0) ** 0.5))

        return np.cumsum(numr, axis=0) / demr

    def _get_sim_ft_pln(self):

        ft = np.zeros(self._sim_shape, dtype=np.complex)

        mag_spec = self._rr.mag_spec.copy()

        if self._sett_init_phs_spec_set_flag:

            if ((self._sett_init_phs_spec_type == 0) or
                (self._alg_rltzn_iter is None)):

                # Outside of annealing or when only one initial phs_spec.
                new_phss = self._sett_init_phs_specs[0]

            elif self._sett_init_phs_spec_type == 1:
                new_phss = self._sett_init_phs_specs[self._alg_rltzn_iter]

            else:
                raise NotImplementedError

            phs_spec = new_phss[1:-1,:].copy()

        else:
            rands = np.random.random((self._sim_shape[0] - 2, 1))

            rands = 1.0 * (-np.pi + (2 * np.pi * rands))

            rands[~self._rr.phs_sel_idxs] = 0.0

            phs_spec = self._rr.phs_spec[1:-1,:].copy()

            phs_spec += rands  # out of bound phs

        mag_spec_flags = None

        ft.real[1:-1,:] = mag_spec[1:-1,:] * np.cos(phs_spec)
        ft.imag[1:-1,:] = mag_spec[1:-1,:] * np.sin(phs_spec)

        self._sim_phs_mod_flags[1:-1,:] += 1

        # First and last coefficients are not written to anywhere, normally.
        ft[+0] = self._rr.ft[+0].copy()
        ft[-1] = self._rr.ft[-1].copy()

        return ft, mag_spec_flags

    def _get_data_ft(self, data, vtype, norm_vals):

        data_ft = np.fft.rfft(data, axis=0)
        data_mag_spec = np.abs(data_ft)[1:-1]

        data_mag_spec = (data_mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (norm_vals is not None):
            data_mag_spec /= norm_vals

        elif (vtype == 'ref') and (norm_vals is None):
            self._rr.data_ft_norm_vals = data_mag_spec[-1,:].copy()

            data_mag_spec /= self._rr.data_ft_norm_vals

        else:
            raise NotImplementedError

        return data_mag_spec

    def _get_probs_ft(self, probs, vtype, norm_vals):

        probs_ft = np.fft.rfft(probs, axis=0)
        probs_mag_spec = np.abs(probs_ft)[1:-1]

        probs_mag_spec = (probs_mag_spec ** 2).cumsum(axis=0)

        if (vtype == 'sim') and (norm_vals is not None):
            probs_mag_spec /= norm_vals

        elif (vtype == 'ref') and (norm_vals is None):
            self._rr.probs_ft_norm_vals = probs_mag_spec[-1,:].copy()

            probs_mag_spec /= self._rr.probs_ft_norm_vals

        else:
            raise NotImplementedError

        return probs_mag_spec

    def _get_gnrc_ft(self, data, vtype):

        assert data.ndim == 1

        ft = np.fft.rfft(data)
        mag_spec = np.abs(ft)

#         mag_spec_sq = mag_spec ** 2

        mag_spec_cumsum = np.concatenate(
            ([ft.real[0]], mag_spec[1:].cumsum()))

        if vtype == 'sim':
            norm_val = None
            sclrs = None
            frst_term = None

        elif vtype == 'ref':
            frst_term = mag_spec_cumsum[0]

            norm_val = float(mag_spec_cumsum[-1])

            mag_spec_cumsum[1:] /= norm_val

            mag_spec_cumsum[:1] = 1

            # sclrs lets the first few long amplitudes into account much
            # better. These describe the direction i.e. Asymmetries.
#             sclrs = 1.0  / np.arange(1.0, mag_spec_cumsum.size + 1.0)
            sclrs = mag_spec / norm_val
            sclrs[0] = 1.0  # sclrs[1:].sum()

        else:
            raise NotImplementedError

        return (mag_spec_cumsum, norm_val, sclrs, frst_term)

    def _get_gnrc_mult_ft(self, data, vtype, tfm_type):

        '''
        IDEA: How about finding the best matching pairs for the reference
        case? And then having individual series. Because, some series might
        not have correlations and result in zero combined variance.

        The ultimate test for this would be to do a rainfall runoff sim
        with the annealed series.
        '''

        data = data.copy(order='f')

        assert tfm_type in ('asymm1', 'asymm2', 'corr', 'etpy')

        assert self._data_ref_n_labels > 1, 'More than one label required!'

        max_comb_size = 2  # self._data_ref_n_labels

        ft_inputs = []

        for comb_size in range(2, max_comb_size + 1):
            combs = combinations(self._data_ref_labels, comb_size)

            n_combs = int(
                factorial(self._data_ref_n_labels) /
                (factorial(comb_size) *
                 factorial(self._data_ref_n_labels - comb_size)))

            input_specs = np.empty(
                ((self._data_ref_shape[0] // 2) + 1, n_combs))

            for i, comb in enumerate(combs):
                col_idxs = [self._data_ref_labels.index(col) for col in comb]

                if len(comb) != 2:
                    raise NotImplementedError('Configured for pairs only!')

                if tfm_type == 'asymm1':
                    # Asymmetry 1.
                    # data is probs.
                    ft_input = (
                        (data[:, col_idxs[0]] +
                         data[:, col_idxs[1]] -
                         1.0) ** asymms_exp)

                elif tfm_type == 'asymm2':
                    # Asymmetry 2.
                    # data is probs.
                    ft_input = (
                        (data[:, col_idxs[0]] -
                         data[:, col_idxs[1]]) ** asymms_exp)

                elif tfm_type == 'corr':
                    ft_input = (
                        (data[:, col_idxs[0]] *
                         data[:, col_idxs[1]]))

                elif tfm_type == 'etpy':
                    ft_input = get_local_entropy_ts_cy(
                        data[:, col_idxs[0]],
                        data[:, col_idxs[1]],
                        self._sett_obj_ecop_dens_bins)

                else:
                    raise NotImplementedError

                ft_inputs.append(ft_input)

                ft = np.fft.rfft(ft_input)
                mag_spec = np.abs(ft)

                # For the multipair case, mag_spec is needed.
                input_spec = mag_spec  # ** 2

                if ft.real[0] < 0:
                    input_spec[0] *= -1

                input_specs[:, i] = input_spec

            break  # Should only happen once due to the pair comb case.

        frst_ft_terms = input_specs[0,:].copy()

        n_frst_ft_terms = frst_ft_terms.size

        # For the single pair case, the power spectrum is taken i.e.
        # its own variance.
        # For the multipair case, magnitude spectrum is taken. This is
        # sort of similar, I think.
        if n_frst_ft_terms == 1:
            input_spec_prod = np.prod(input_specs[1:,:] ** 2, axis=1)

        else:
            input_spec_prod = np.prod(input_specs[1:,:], axis=1)

        input_spec_cumsum = np.concatenate(
            (frst_ft_terms, input_spec_prod.cumsum()))

        if vtype == 'sim':
            norm_val = None
            sclrs = None
            frst_ft_terms = None

        elif vtype == 'ref':
            if n_frst_ft_terms == 1:
                norm_val = np.prod(
                    (input_specs[1:,:] ** 2).sum(axis=0) ** 2) ** 0.5

            else:
                norm_val = np.prod(
                    (input_specs[1:,:] ** n_frst_ft_terms
                    ).sum(axis=0)) ** (1 / n_frst_ft_terms)

            input_spec_cumsum[n_frst_ft_terms:] /= norm_val

            input_spec_cumsum[:n_frst_ft_terms] = 1

            # sclrs lets the first few long amplitudes into account much
            # better. These describe the direction i.e. Asymmetries.
            sclrs = 1.0 / np.arange(1.0, input_spec_cumsum.size + 1.0)
            sclrs[:n_frst_ft_terms] = 1.0  # input_spec_cumsum.size / n_frst_ft_terms

        else:
            raise NotImplementedError

        return (
            input_spec_cumsum,
            norm_val,
            sclrs,
            n_frst_ft_terms,
            frst_ft_terms)

