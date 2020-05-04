'''
@author: Faizan

Dec 30, 2019

1:25:46 PM
'''

from .save import PhaseAnnealingSave as PAS


class PhaseAnnealing(PAS):

    '''
    Phase Annealing

    Description
    -----------
    The phase annealing algorithm to generate a 1D series at N points
    that have some given prescribed properties similar to a reference series.

    Usage
    -----
    Call the following methods in the given order. All are required except
    when mentioned otherwise. See their respective documentations for the
    form of inputs. For usage you can also refer to test_phsann.py in the
    test directory.

    01. set_reference_data
    02. set_objective_settings
    03. set_annealing_settings
    04. set_annealing_auto_temperature_settings (optional)
    05. set_extended_length_sim_settings (optional)
    06. set_mult_phase_settings (optional)
    07. set_objective_weights_settings (optional)
    08. set_misc_settings
    09. prepare
    10. verify
    11. simulate

    Outputs
    -------
    All outputs are saved to the HDF5 file phsann.h5 in the outputs directory.
    '''

    def __init__(self, verbose=True):

        PAS.__init__(self, verbose)

        self._main_verify_flag = False
        return

    def verify(self):

        PAS._PhaseAnnealingSave__verify(self)
        assert self._save_verify_flag, 'Save in an unverified state!'

        self._main_verify_flag = True
        return
