'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from time import asctime
from timeit import default_timer

import h5py
import numpy as np
from multiprocessing import Manager, Lock
from pathos.multiprocessing import ProcessPool

from ...misc import print_sl, print_el, ret_mp_idxs
from ..prepare import PhaseAnnealingPrepare as PAP

from .obj import PhaseAnnealingAlgObjective
from .aio import PhaseAnnealingAlgIO
from .lagnthwts import PhaseAnnealingAlgLagNthWts
from .labelwts import PhaseAnnealingAlgLabelWts
from .autoobjwts import PhaseAnnealingAlgAutoObjWts
from .rltzn import PhaseAnnealingAlgRealization
from .tem import PhaseAnnealingAlgTemperature
from .misca import PhaseAnnealingAlgMisc


class PhaseAnnealingAlgorithm(
        PAP,
        PhaseAnnealingAlgObjective,
        PhaseAnnealingAlgIO,
        PhaseAnnealingAlgLagNthWts,
        PhaseAnnealingAlgLabelWts,
        PhaseAnnealingAlgAutoObjWts,
        PhaseAnnealingAlgRealization,
        PhaseAnnealingAlgTemperature,
        PhaseAnnealingAlgMisc):

    '''The main phase annealing class.'''

    def __init__(self, verbose=True):

        # Instance variables defined here are used in the classes that this
        # inherits from. It does create the problem of looking for their
        # definition. But resolving this problems leads to creation of more.

        PAP.__init__(self, verbose)

        self._lock = None

        self._alg_rltzn_iter = None

        # Snapshot.
        self._alg_snapshot = None

        # Lag/Nth  weights.
        self._alg_wts_lag_nth_search_flag = False
        self._alg_wts_lag_scorr = None
        self._alg_wts_lag_asymm_1 = None
        self._alg_wts_lag_asymm_2 = None
        self._alg_wts_lag_ecop_dens = None
        self._alg_wts_lag_ecop_etpy = None
        self._alg_wts_nth_order = None
        self._alg_wts_lag_pcorr = None
        self._alg_wts_lag_asymm_1_ft = None
        self._alg_wts_lag_asymm_2_ft = None
        self._alg_wts_nth_order_ft = None
        self._alg_wts_lag_etpy_ft = None

        # Label  weights.
        self._alg_wts_label_search_flag = False
        self._alg_wts_label_scorr = None
        self._alg_wts_label_asymm_1 = None
        self._alg_wts_label_asymm_2 = None
        self._alg_wts_label_ecop_dens = None
        self._alg_wts_label_ecop_etpy = None
        self._alg_wts_label_nth_order = None
        self._alg_wts_label_pcorr = None
        self._alg_wts_label_asymm_1_ft = None
        self._alg_wts_label_asymm_2_ft = None
        self._alg_wts_label_nth_order_ft = None
        self._alg_wts_label_etpy_ft = None

        # Obj wts.
        self._alg_wts_obj_search_flag = False
        self._alg_wts_obj_raw = None

        # Stopping criteria labels.
        self._alg_cnsts_stp_crit_labs = (
            'Iteration completion',
            'Iterations without acceptance',
            'Running objective function tolerance',
            'Annealing temperature',
            'Running phase reduction rate',
            'Running acceptance rate',
            'Iterations without updating the global minimum')

        # Closest value that is less than or equal to be taken as a zero.
        self._alg_cnsts_almost_zero = 1e-15

        # Minimum phase reduction rate beyond which the rate is taken as zero.
        self._alg_cnsts_min_phs_red_rate = 1e-4

        # Exponent to take before computing sum of differences in
        # objective functions. Should be an even value.
        self._alg_cnsts_diffs_exp = 2.0

        # Limiting values of minimum and maximum probabilites in the
        # objective functions when simulation probabilities are computed
        # inversly. This limits the cases when the extrapolation goes way
        # below or beyond, 0 and 1 respectively.
        self._alg_cnsts_min_prob_val = -0.1
        self._alg_cnsts_max_prob_val = +1.1

        # A flag to tell if to use non-exceedence probabilities for computing
        # the objective values or the histogram. This is used along with
        # other criteria. So, the effect can be different when the flag is
        # set or unset.
        self._alg_cnsts_lag_wts_overall_err_flag = True

        # Flags.
        self._alg_rltzns_gen_flag = False
        self._alg_force_acpt_flag = False
        self._alg_done_opt_flag = False
        self._alg_ann_runn_auto_init_temp_search_flag = False
        self._alg_verify_flag = False
        return

    def simulate(self):

        '''Start the phase annealing algorithm'''

        assert self._alg_verify_flag, 'Call verify first!'

        beg_sim_tm = default_timer()

        self._init_output()

        if self._sett_auto_temp_set_flag:
            self._sett_misc_auto_init_temp_dir.mkdir(exist_ok=True)

            assert self._sett_misc_auto_init_temp_dir.exists(), (
                'Could not create auto_init_temp_dir!')

        self._update_wts(1.0, 1.0)

        if self._sett_auto_temp_set_flag:
            self._search_init_temp()

        if self._vb:
            print_sl()

            temp = self._sett_ann_init_temp
            assert temp > self._alg_cnsts_almost_zero, (
                'Initial temperature almost zero!')

            ctr = 0
            while True:
                ctr += self._sett_ann_upt_evry_iter
                temp *= self._sett_ann_temp_red_rate

                if temp <= self._alg_cnsts_almost_zero:
                    break

            print(
                f'Maximum number of iterations possible with the set initial '
                f'temperature ({self._sett_ann_init_temp:5.3E}) are '
                f'{ctr:1.1E}.')

            if self._sett_ann_max_iters > ctr:
                print(
                    f'Warning: set maximum number of iterations '
                    f'({self._sett_ann_max_iters:1.1E}) unreachable with '
                    f'this initial temperature!')

                self._sett_ann_max_iters = ctr

                print(
                    f'Reset maximum number of iterations to: '
                    f'{self._sett_ann_max_iters:1.1E}')

            print_el()

        self._write_non_sim_data_to_h5()

        if self._vb:
            print_sl()

            print('Starting simulations...')

            print_el()

            print('\n')

        n_cpus = min(self._sett_misc_n_rltzns, self._sett_misc_n_cpus)

        if n_cpus > 1:

            mp_idxs = ret_mp_idxs(self._sett_misc_n_rltzns, n_cpus)

            rltzns_gen = (
                (
                (mp_idxs[i], mp_idxs[i + 1]),
                )
                for i in range(mp_idxs.size - 1))

            self._lock = Manager().Lock()

            mp_pool = ProcessPool(n_cpus)
            mp_pool.restart(True)

            list(mp_pool.uimap(self._sim_grp, rltzns_gen))

            mp_pool.close()
            mp_pool.join()

            self._lock = None

            mp_pool = None

        else:
            self._lock = Lock()

            self._sim_grp(((0, self._sett_misc_n_rltzns),))

            self._lock = None

        end_sim_tm = default_timer()

        if self._vb:
            print_sl()

            print(
                f'Total simulation time was: '
                f'{end_sim_tm - beg_sim_tm:0.3f} seconds!')

            print_el()

        self._alg_rltzns_gen_flag = True
        return

    def verify(self):

        PAP._PhaseAnnealingPrepare__verify(self)
        assert self._prep_verify_flag, 'Prepare in an unverified state!'

        if self._vb:
            print_sl()

            print(
                'Phase annealing algorithm requirements verified '
                'successfully!')

            print_el()

        self._alg_verify_flag = True
        return

    def _init_output(self):

        if self._vb:
            print_sl()

            print('Initializing outputs file...')

        if not self._sett_misc_outs_dir.exists():
            self._sett_misc_outs_dir.mkdir(exist_ok=True)

        assert self._sett_misc_outs_dir.exists(), (
            'Could not create outputs_dir!')

        h5_path = self._sett_misc_outs_dir / self._save_h5_name

        # Create new / overwrite old.
        # It's simpler to do it here.
        h5_hdl = h5py.File(h5_path, mode='w', driver=None)
        h5_hdl.close()

        if self._vb:
            print('Initialized the outputs file.')

            print_el()
        return

    def _sim_grp(self, args):

        ((beg_rltzn_iter, end_rltzn_iter),
        ) = args

        beg_thread_time = default_timer()

        for rltzn_iter in range(beg_rltzn_iter, end_rltzn_iter):

            self._reset_timers()

            self._gen_ref_aux_data()

            if self._vb:
                with self._lock:
                    print(
                        f'Started realization {rltzn_iter} on {asctime()}...')

            beg_rltzn_tm = default_timer()

            stopp_criteria = self._gen_gnrc_rltzn(
                (rltzn_iter, None))

            end_rltzn_tm = default_timer()

            if self._vb:
                with self._lock:

                    print('\n')

                    assert len(stopp_criteria) == len(
                        self._alg_cnsts_stp_crit_labs)

                    stopp_criteria_labels_rltzn = [
                        self._alg_cnsts_stp_crit_labs[i]
                        for i in range(len(stopp_criteria))
                        if not stopp_criteria[i]]

                    assert len(stopp_criteria_labels_rltzn), (
                        'No stopp_criteria!')

                    stopp_criteria_str = ' and '.join(
                        stopp_criteria_labels_rltzn)

                    print(
                        f'Realization {rltzn_iter} took '
                        f'{end_rltzn_tm - beg_rltzn_tm:0.3f} '
                        f'seconds with stopp_criteria: '
                        f'{stopp_criteria_str}.')

            self._reset_timers()

            assert np.all(self._sim_phs_mod_flags >= 1), (
                'Some phases were not modified!')

        end_thread_time = default_timer()

        if self._vb:
            with self._lock:
                print(
                    f'Total thread time for realizations between '
                    f'{beg_rltzn_iter} and {end_rltzn_iter - 1} was '
                    f'{end_thread_time - beg_thread_time:0.3f} '
                    f'seconds.')
        return

    __verify = verify
