'''
Created on Nov 16, 2021

@author: Faizan3800X-Uni
'''

from .rltzngnrc import PhaseAnnealingPrepareRltznGnrc as PRG


class PhaseAnnealingPrepareRltznSim(PRG):

    def __init__(self):

        PRG.__init__(self)

        # Add var labs to _getdata in save.py if they need to be there.

        # self.label = None
        # self.probs = None
        # self.ft = None
        # self.phs_spec = None
        # self.mag_spec = None
        # self.scorrs = None
        # self.asymms_1 = None
        # self.asymms_2 = None
        # self.ecop_dens = None
        # self.ecop_etpy = None
        # self.data = None
        # self.pcorrs = None
        # self.nths = None
        # self.data_ft = None
        # self.probs_ft = None

        self.shape = None
        self.mag_spec_cdf = None
        self.ft_best = None

        # To keep track of modified phases.
        self.phs_mod_flags = None
        self.n_idxs_all_cts = None
        self.n_idxs_acpt_cts = None

        # An array. False for phase changes, True for coeff changes.
        self.mag_spec_flags = None

        # Objective function variables.
        self.scorr_diffs = None
        self.asymm_1_diffs = None
        self.asymm_2_diffs = None
        self.ecop_dens_diffs = None
        self.ecop_etpy_diffs = None
        self.nth_ord_diffs = None
        self.pcorr_diffs = None

        self.mult_asymms_1_diffs = None
        self.mult_asymms_2_diffs = None
        self.mult_ecop_dens = None

        self.asymm_1_diffs_ft = None
        self.asymm_2_diffs_ft = None
        self.nth_ord_diffs_ft = None
        self.etpy_ft = None
        self.mult_asymm_1_cmpos_ft = None
        self.mult_asymm_2_cmpos_ft = None
        self.mult_etpy_cmpos_ft = None

        # QQ probs
        self.scorr_qq_dict = None
        self.asymm_1_qq_dict = None
        self.asymm_2_qq_dict = None
        self.ecop_dens_qq_dict = None
        self.ecop_etpy_qq_dict = None
        self.nth_ords_qq_dict = None
        self.pcorr_qq_dict = None

        self.mult_asymm_1_qq_dict = None
        self.mult_asymm_2_qq_dict = None
        self.mult_ecop_dens_qq_dict = None  # TODO

        # Misc.
        self.mag_spec_idxs = None
        self.rltzns_proto_tup = None

        # Durations.
        self.cumm_call_durations = None
        self.cumm_n_calls = None
        return
