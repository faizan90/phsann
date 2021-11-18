'''
Created on Nov 17, 2021

@author: Faizan3800X-Uni
'''
from ..settings import PhaseAnnealingSettings as PAS
from .rltznref import PhaseAnnealingPrepareRltznRef as PRR
from .rltznsim import PhaseAnnealingPrepareRltznSim as PRS


class PhaseAnnealingPrepareBase(PAS):

    def __init__(self, verbose=True):

        PAS.__init__(self, verbose)

        self._rr = PRR()  # Reference.
        self._rs = PRS()  # Simulation.

        self._data_tfm_type = 'probs'
        self._data_tfm_types = (
            'log_data', 'probs', 'data', 'probs_sqrt', 'norm')

        # Flags.
        self._prep_ref_aux_flag = False
        self._prep_sim_aux_flag = False
        self._prep_prep_flag = False
        self._prep_verify_flag = False

        # Validation steps.
        self._prep_vld_flag = False
        return
