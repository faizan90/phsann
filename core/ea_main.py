'''
Created on Dec 29, 2021

@author: Faizan3800X-Uni
'''

from gnrctsgenr import (
    GTGBase,
    GTGData,
    GTGPrepareBase,
    GTGPrepareCDFS,
    GTGPrepareUpdate,
    GTGAlgBase,
    GTGAlgObjective,
    GTGAlgIO,
    GTGAlgTemperature,
    GTGAlgMisc,
    GTGAlgorithm,
    GTGSave,
    )

from .aa_setts import PhaseAnnealingSettings

from .ba_prep import (
    PhaseAnnealingPrepareRltznRef,
    PhaseAnnealingPrepareRltznSim,
    PhaseAnnealingPrepareTfms,
    PhaseAnnealingPrepare,
    )

from .ca_alg import (
    PhaseAnnealingAlgLagNthWts,
    PhaseAnnealingAlgLabelWts,
    PhaseAnnealingAlgAutoObjWts,
    )

from .da_rltzn import PhaseAnnealingRealization


class PhaseAnnealingMain(
        GTGBase,
        GTGData,
        PhaseAnnealingSettings,
        GTGPrepareBase,
        PhaseAnnealingPrepareTfms,
        GTGPrepareCDFS,
        GTGPrepareUpdate,
        PhaseAnnealingPrepare,
        GTGAlgBase,
        GTGAlgObjective,
        GTGAlgIO,
        PhaseAnnealingAlgLagNthWts,
        PhaseAnnealingAlgLabelWts,
        PhaseAnnealingAlgAutoObjWts,
        PhaseAnnealingRealization,
        GTGAlgTemperature,
        GTGAlgMisc,
        GTGAlgorithm,
        GTGSave):

    def __init__(self, verbose):

        GTGBase.__init__(self, verbose)
        GTGData.__init__(self)
        PhaseAnnealingSettings.__init__(self)

        self._rr = PhaseAnnealingPrepareRltznRef()  # Reference.
        self._rs = PhaseAnnealingPrepareRltznSim()  # Simulation.

        GTGPrepareBase.__init__(self)
        PhaseAnnealingPrepareTfms
        GTGPrepareCDFS
        GTGPrepareUpdate
        PhaseAnnealingPrepare
        GTGAlgBase.__init__(self)
        GTGAlgObjective.__init__(self)
        GTGAlgIO.__init__(self)
        PhaseAnnealingAlgLagNthWts.__init__(self)
        PhaseAnnealingAlgLabelWts.__init__(self)
        PhaseAnnealingAlgAutoObjWts.__init__(self)
        PhaseAnnealingRealization.__init__(self)
        GTGAlgTemperature.__init__(self)
        GTGAlgMisc.__init__(self)
        GTGAlgorithm.__init__(self)
        GTGSave.__init__(self)

        self._main_verify_flag = False
        return

    def verify(self):

        GTGData._GTGData__verify(self)

        PhaseAnnealingSettings._PhaseAnnealingSettings__verify(self)

        assert self._sett_ann_pa_sa_sett_verify_flag, (
            'Phase Aneealing settings in an unverfied state!')

        PhaseAnnealingPrepare._PhaseAnnealingPrepare__verify(self)
        GTGAlgorithm._GTGAlgorithm__verify(self)
        GTGSave._GTGSave__verify(self)

        assert self._save_verify_flag, 'Save in an unverified state!'

        self._main_verify_flag = True
        return
