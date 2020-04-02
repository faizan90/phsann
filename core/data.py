'''
Created on Dec 27, 2019

@author: Faizan
'''

import numpy as np

from ..misc import print_sl, print_el

eps_err_flag = True
eps_err = 1e-7


class PhaseAnnealingData:

    '''Set the reference data'''

    def __init__(self, verbose=True):

        '''
        Parameters
        ----------
        verbose : bool
            Whether to show activity messages
        '''

        assert isinstance(verbose, bool), 'verbose not a boolean!'

        self._vb = verbose

        self._data_ref_rltzn = None
        self._data_ref_shape = None
        self._data_ref_rltzn_srtd = None
        self._data_ref_labels = None
        self._data_ref_n_labels = None

        self._data_min_pts = 3

        self._data_ref_set_flag = False
        self._data_verify_flag = False
        return

    def set_reference_data(self, ref_rltzn, labels=None):

        '''
        Set the reference data array

        Parameters
        ----------
        ref_rltzn : 2D float64 np.ndarray
            The reference realization/data array. No NaNs or Infinitys allowed.
            Rows are steps and columns are stations.
        labels : list or None
            Labels used to address each column. Will be cast to strings. All
            must be unique.
            If None, a 0-indexed labeling is used.
        '''

        if self._vb:
            print_sl()

            print('Setting reference data for phase annealing...\n')

        assert isinstance(ref_rltzn, np.ndarray), (
            'ref_rltzn not a numpy array!')

        assert ref_rltzn.ndim == 2, 'ref_rltzn not a 2D array!'
        assert np.all(np.isfinite(ref_rltzn)), 'Invalid values in ref_rltzn!'
        assert ref_rltzn.dtype == np.float64, 'ref_rltzn dtype not np.float64!'

        assert isinstance(labels, (type(None), list)), (
            'labels neither list nor None!')

        if labels is None:
            labels = list(range(ref_rltzn.shape[1]))

        assert len(labels) == np.unique(labels).size, (
            'Non unique labels!')

        labels = tuple([str(label) for label in labels])

        assert len(labels) == ref_rltzn.shape[1], (
            'Number of labels and columns in ref_rltzn of unequal length!')

        if ref_rltzn.shape[0] % 2:
            ref_rltzn = ref_rltzn[:-1]

            print('Warning: dropped last step for even steps!\n')

        assert 0 < self._data_min_pts <= ref_rltzn.shape[0], (
            'ref_rltzn has too few steps!')

        if eps_err_flag:
            # TODO: precipitation-type functions will have to be dealt with
            # here differerntly.
            for i in range(ref_rltzn.shape[1]):
                unq_vals = np.unique(ref_rltzn[:, i])

                if unq_vals.size == ref_rltzn.shape[0]:
                    continue

                eps_errs = -eps_err + (
                    2 * eps_err * np.random.random(ref_rltzn.shape[0]))

                ref_vals_eps = ref_rltzn[:, i] + eps_errs

                assert np.unique(ref_vals_eps).size == ref_rltzn.shape[0], (
                    f'Non-unique values in reference for label '
                    f'{labels[i]} after adding eps_err!')

                ref_rltzn[:, i] = ref_vals_eps

        self._data_ref_rltzn = ref_rltzn
        self._data_ref_shape = ref_rltzn.shape

        self._data_ref_labels = labels
        self._data_ref_n_labels = len(labels)

        self._data_ref_rltzn_srtd = np.sort(self._data_ref_rltzn, axis=0)

        if self._vb:
            print(
                f'Reference realization set with shape: '
                f'{self._data_ref_shape}')

            print(
                f'Labels: {labels}')

            print_el()

        self._data_ref_set_flag = True
        return

    def verify(self):

        '''Verify if data has been set correctly'''

        assert self._data_ref_set_flag, 'Call set_reference_data first!'

        if self._vb:
            print_sl()

            print(f'Phase annealing data verified successfully!')

            print_el()

        self._data_verify_flag = True
        return

    __verify = verify
