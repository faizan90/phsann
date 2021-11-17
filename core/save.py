'''
@author: Faizan-Uni

Jan 16, 2020

9:39:36 AM
'''
from pathlib import Path
from fnmatch import fnmatch

import h5py
import numpy as np
#
from ..misc import print_sl, print_el

from .algorithm import PhaseAnnealingAlgorithm as PAA


class PhaseAnnealingSave(PAA):

    '''
    Save reference, realizations flags and settings to HDF5
    '''

    def __init__(self, verbose):

        PAA.__init__(self, verbose)

        self._save_h5_name = 'phsann.h5'

        self._save_verify_flag = True
        return

    def _get_flags(self):

        flags = [('vb', self._vb)]
        for var in vars(self):
            if not fnmatch(var, '*_flag'):
                continue

            flag = getattr(self, var)

            if not isinstance(flag, bool):
                continue

            flags.append((var.lstrip('_'), flag))

        assert flags, 'No flags selected!'

        return flags

    def _write_flags(self, h5_hdl):

        # Don't use underscores in names as they are lstripped
        # in the get method.

        flags = self._get_flags()

        flags_grp = h5_hdl.create_group('flags')

        for flag_lab, flag_val in flags:
            flags_grp.attrs[flag_lab] = flag_val

        h5_hdl.flush()
        return

    def _get_ref_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_data_*'):
                continue

            datas.append((var.lstrip('_'), getattr(self, var)))

        return datas

    def _write_ref_data(self, h5_hdl):

        # Don't use underscores in names as they are lstripped
        # in the get method.

        datas = self._get_ref_data()

        datas_grp = h5_hdl.create_group('data_ref')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                datas_grp[data_lab] = data_val

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    def _get_settings(self):

        setts = []
        for var in vars(self):
            if not fnmatch(var, '_sett_*'):
                continue

            setts.append((var.lstrip('_'), getattr(self, var)))

        return setts

    def _write_settings(self, h5_hdl):

        # Don't use underscores in names as they are lstripped
        # in the get method.

        setts = self._get_settings()

        setts_grp = h5_hdl.create_group('settings')

        for sett_lab, sett_val in setts:
            if isinstance(sett_val, np.ndarray):

                if sett_lab == 'sett_obj_flag_labels':
                    dt = h5py.special_dtype(vlen=str)

                    tre = setts_grp.create_dataset(
                        sett_lab, (sett_val.shape[0],), dtype=dt)

                    tre[:] = sett_val

                else:
                    setts_grp[sett_lab] = sett_val

            elif sett_val is None:
                setts_grp.attrs[sett_lab] = str(sett_val)

            elif isinstance(sett_val, Path):
                setts_grp.attrs[sett_lab] = str(sett_val)

            else:
                setts_grp.attrs[sett_lab] = sett_val

        h5_hdl.flush()
        return

    def _get_prep_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_prep_*'):
                continue

            datas.append((var.lstrip('_'), getattr(self, var)))

        return datas

    def _write_prep_data(self, h5_hdl):

        # Don't use underscores in names as they are lstripped
        # in the get method.

        datas = self._get_prep_data()

        datas_grp = h5_hdl.create_group('prep')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                datas_grp[data_lab] = data_val

            elif data_val is None:
                datas_grp.attrs[data_lab] = str(data_val)

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    def _get_alg_data(self):

        datas = []
        for var in vars(self):
            if not fnmatch(var, '_alg_*'):
                continue

            if var in ('_alg_rltzns',):
                continue

            datas.append((var.lstrip('_'), getattr(self, var)))

        return datas

    def _write_alg_data(self, h5_hdl):

        # Don't use underscores in names as they are lstripped
        # in the get method.

        datas = self._get_alg_data()

        datas_grp = h5_hdl.create_group('algorithm')

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                if data_lab in ('alg_auto_temp_search_ress',):

                    pad_zeros = len(str(self._sett_misc_n_rltzns))

                    grp = datas_grp.create_group(f'{data_lab}')

                    for i in range(data_val.shape[0]):
                        grp[f'{i:0{pad_zeros}d}'] = data_val[i]

                else:
                    datas_grp[data_lab] = data_val

            elif data_val is None:
                datas_grp.attrs[data_lab] = str(data_val)

            elif ((isinstance(data_val, dict) and
                  (fnmatch(data_lab, 'alg_wts*')))):

                wts_grp = datas_grp.create_group(data_lab)
                for key in data_val:
                    wts_grp.attrs[str(key)] = data_val[key]

            else:
                datas_grp.attrs[data_lab] = data_val

        h5_hdl.flush()
        return

    # def _get_sim_data(self):
    #
    #     sim_var_labs = [
    #         '_sim_shape',
    #         ]
    #
    #     datas = []
    #     for var in vars(self):
    #         if var not in sim_var_labs:
    #             continue
    #
    #         datas.append((var.lstrip('_'), getattr(self, var)))
    #
    #     return datas
    #
    # def _write_sim_data(self, h5_hdl):
    #
    #     # Don't use underscores in names as they are lstripped
    #     # in the get method.
    #
    #     datas = self._get_sim_data()
    #
    #     datas_grp = h5_hdl.create_group('data_sim')
    #
    #     for data_lab, data_val in datas:
    #         if isinstance(data_val, np.ndarray):
    #             datas_grp[data_lab] = data_val
    #
    #         elif data_val is None:
    #             datas_grp.attrs[data_lab] = str(data_val)
    #
    #         else:
    #             datas_grp.attrs[data_lab] = data_val
    #
    #     h5_hdl.flush()
    #     return

    def _write_non_sim_data_to_h5(self):

        if self._vb:
            print_sl()

            print('Writing non-simulation data to HDF5...')

        self._sett_misc_outs_dir.mkdir(exist_ok=True)

        assert self._sett_misc_outs_dir.exists(), (
            'Could not create outputs_dir!')

        h5_path = self._sett_misc_outs_dir / self._save_h5_name

        with h5py.File(h5_path, mode='a', driver=None) as h5_hdl:

            self._write_flags(h5_hdl)

            self._write_ref_data(h5_hdl)

            self._write_settings(h5_hdl)

            self._write_prep_data(h5_hdl)

            self._write_alg_data(h5_hdl)

            # self._write_sim_data(h5_hdl)

        if self._vb:
            print('Done writing.')

            print_el()

        return

    def update_h5_file_name(self, new_name):

        if self._vb:
            print_sl()

            print('Updating output HDF5 file name...')

        assert isinstance(new_name, str), 'new_name not a string!'

        assert new_name, 'new_name is empty!'
        assert '.' in new_name, 'new_name has no extension!'

        old_name = self._save_h5_name

        self._save_h5_name = new_name

        if self._vb:
            print(f'Changed name from {old_name} to {self._save_h5_name}!')

            print_el()

        return

    def get_h5_file_path(self):

        assert self._save_verify_flag, 'Save in an unverified state!'

        return self._sett_misc_outs_dir / self._save_h5_name

    def verify(self):

        PAA._PhaseAnnealingAlgorithm__verify(self)
        assert self._alg_verify_flag, 'Algorithm in an unverified state!'

        self._save_verify_flag = True
        return

    __verify = verify
