'''
Created on Nov 15, 2021

@author: Faizan3800X-Uni
'''

from fnmatch import fnmatch

import h5py
import numpy as np
from scipy.interpolate import interp1d


class PhaseAnnealingAlgIO:

    '''
    Supporting class of Algorithm.

    Has no verify method or any private variables of its own.
    '''

    def _write_cls_rltzn(self, rltzn_iter, ret):

        with self._lock:
            h5_path = self._sett_misc_outs_dir / self._save_h5_name

            with h5py.File(h5_path, mode='a', driver=None) as h5_hdl:
                self._write_ref_rltzn(h5_hdl)
                self._write_sim_rltzn(h5_hdl, rltzn_iter, ret)
        return

    def _write_ref_rltzn(self, h5_hdl):

        # Should be called by _write_rltzn with a lock.

        ref_grp_lab = 'data_ref_rltzn'

        if ref_grp_lab in h5_hdl:
            return

        datas = []
        for var in vars(self._rr):
            # if not fnmatch(var, '_ref_*'):
            #     continue

            datas.append((var, getattr(self._rr, var)))

        ref_grp = h5_hdl.create_group(ref_grp_lab)

        ll_idx = 0  # ll is for label.
        lg_idx = 1  # lg is for lag.

        for data_lab, data_val in datas:
            if isinstance(data_val, np.ndarray):
                ref_grp[data_lab] = data_val

            elif isinstance(data_val, interp1d):
                ref_grp[data_lab + '_x'] = data_val.x
                ref_grp[data_lab + '_y'] = data_val.y

            # Single obj. vals. dicts.
            elif (isinstance(data_val, dict) and

                  all([isinstance(key[lg_idx], np.int64)
                       for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'

                    ref_grp[data_lab + f'{lab}_x'] = data_val[key].xr
                    ref_grp[data_lab + f'{lab}_y'] = data_val[key].yr

            elif (isinstance(data_val, dict) and

                  all([key[lg_idx] in ('cos', 'sin') for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]}'
                    ref_grp[data_lab + f'{lab}_x'] = data_val[key].x
                    ref_grp[data_lab + f'{lab}_y'] = data_val[key].y

            elif (isinstance(data_val, dict) and

                  all([isinstance(key[lg_idx], np.int64)
                       for key in data_val]) and

                  all([isinstance(val, np.ndarray)
                       for val in data_val.values()])):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'
                    ref_grp[data_lab + lab] = data_val[key]

            # Multsite obj. vals. dicts.
            elif (isinstance(data_val, dict) and

                  all([all([col in self._data_ref_labels for col in key])
                       for key in data_val]) and

                  all([isinstance(val, interp1d)
                       for val in data_val.values()])):

                for key in data_val:
                    comb_str = '_'.join(key)
                    ref_grp[f'{data_lab}_{comb_str}_x'] = data_val[key].xr
                    ref_grp[f'{data_lab}_{comb_str}_y'] = data_val[key].yr

            # For mult site ecop stuff.
            elif (isinstance(data_val, dict) and

                 all([isinstance(data_val[key], np.ndarray)
                      for key in data_val]) and

                 fnmatch(data_lab, '*mult_ecop_dens_diffs_cdfs*')):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]}'
                    ref_grp[data_lab + f'{lab}'] = data_val[key]

            elif isinstance(data_val, (str, float, int)):
                ref_grp.attrs[data_lab] = data_val

            elif data_val is None:
                ref_grp.attrs[data_lab] = str(data_val)

            elif (isinstance(data_val, dict) and

                 all([isinstance(data_val[key], np.ndarray)
                      for key in data_val]) and

                 fnmatch(data_lab, '*_qq_*')):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]}'
                    ref_grp[data_lab + f'{lab}'] = data_val[key]

            # For diff fts dicts.
            elif (isinstance(data_val, dict) and

                 all([isinstance(
                     data_val[key], tuple) for key in data_val]) and

                 fnmatch(data_lab, '*_diffs_ft*')):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'
                    ref_grp[data_lab + f'{lab}'] = data_val[key][0]

            # For etpy fts dicts.
            elif (isinstance(data_val, dict) and

                 all([isinstance(
                     data_val[key], tuple) for key in data_val]) and

                 fnmatch(data_lab, '*etpy_ft*')):

                for key in data_val:
                    lab = f'_{key[ll_idx]}_{key[lg_idx]:03d}'
                    ref_grp[data_lab + f'{lab}'] = data_val[key][0]

            # For mult cmpos fts dicts.
            elif (isinstance(data_val, tuple) and

                 (len(data_val) == 5) and

                 fnmatch(data_lab, 'mult_*_cmpos_ft_*')):

                ref_grp[data_lab] = data_val[0]

            else:
                raise NotImplementedError(
                    f'Unknown type {type(data_val)} for variable '
                    f'{data_lab}!')

        h5_hdl.flush()
        return

    def _write_sim_rltzn(self, h5_hdl, rltzn_iter, ret):

        # Should be called by _write_rltzn with a lock

        sim_pad_zeros = len(str(self._sett_misc_n_rltzns))

        main_sim_grp_lab = 'data_sim_rltzns'

        sim_grp_lab = f'{rltzn_iter:0{sim_pad_zeros}d}'

        if not main_sim_grp_lab in h5_hdl:
            sim_grp_main = h5_hdl.create_group(main_sim_grp_lab)

        else:
            sim_grp_main = h5_hdl[main_sim_grp_lab]

        if not sim_grp_lab in sim_grp_main:
            sim_grp = sim_grp_main.create_group(sim_grp_lab)

        else:
            sim_grp = sim_grp_main[sim_grp_lab]

        for lab, val in ret._asdict().items():
            if isinstance(val, np.ndarray):
                sim_grp[lab] = val

            elif fnmatch(lab, 'tmr*') and isinstance(val, dict):
                tmr_grp = sim_grp.create_group(lab)
                for meth_name, meth_val in val.items():
                    tmr_grp.attrs[meth_name] = meth_val

            else:
                sim_grp.attrs[lab] = val

        if self._sim_mag_spec_flags is not None:
            sim_grp['sim_mag_spec_flags'] = (
                self._sim_mag_spec_flags)

        if self._sim_mag_spec_idxs is not None:
            sim_grp['sim_mag_spec_idxs'] = self._sim_mag_spec_idxs

        h5_hdl.flush()
        return
