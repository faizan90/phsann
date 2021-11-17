'''
Created on Nov 17, 2021

@author: Faizan3800X-Uni
'''

from ..prepare import PhaseAnnealingPrepare as PAP


class PhaseAnnealingAlgBase(PAP):

    def __init__(self, verbose=True):

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
        self._sett_wts_obj_wts = None
        self._alg_wts_obj_raw = None
        self._alg_wts_obj_search_flag = False

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
