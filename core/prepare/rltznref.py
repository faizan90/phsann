'''
Created on Nov 16, 2021

@author: Faizan3800X-Uni
'''

from .rltzngnrc import PhaseAnnealingPrepareRltznGnrc as PRG


class PhaseAnnealingPrepareRltznRef(PRG):

    def __init__(self):

        PRG.__init__(self)

        self.label = 'ref'

        self.ft_cumm_corr = None
        self.probs_srtd = None
        self.data_ft_norm_vals = None
        self.data_tfm = None
        self.phs_sel_idxs = None
        self.phs_idxs = None
        self.probs_ft_norm_vals = None

        # Reference data for objective functions.
        self.scorr_diffs_cdfs_dict = None
        self.asymm_1_diffs_cdfs_dict = None
        self.asymm_2_diffs_cdfs_dict = None
        self.ecop_dens_diffs_cdfs_dict = None
        self.ecop_etpy_diffs_cdfs_dict = None
        self.nth_ord_diffs_cdfs_dict = None
        self.pcorr_diffs_cdfs_dict = None
        self.cos_sin_cdfs_dict = None

        self.mult_asymm_1_diffs_cdfs_dict = None
        self.mult_asymm_2_diffs_cdfs_dict = None
        self.mult_ecop_dens_cdfs_dict = None

        self.scorr_qq_dict = None
        self.asymm_1_qq_dict = None
        self.asymm_2_qq_dict = None
        self.ecop_dens_qq_dict = None
        self.ecop_etpy_qq_dict = None
        self.nth_ord_qq_dict = None
        self.pcorr_qq_dict = None

        self.mult_asymm_1_qq_dict = None
        self.mult_asymm_2_qq_dict = None
        self.mult_etpy_dens_qq_dict = None

        self.asymm_1_diffs_ft_dict = None
        self.asymm_2_diffs_ft_dict = None
        self.nth_ord_diffs_ft_dict = None
        self.etpy_ft_dict = None
        self.mult_asymm_1_cmpos_ft_dict = None
        self.mult_asymm_2_cmpos_ft_dict = None
        self.mult_etpy_cmpos_ft_dict = None
        return
