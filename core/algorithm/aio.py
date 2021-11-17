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

    def _write_cls_rltzn(self):

        with self._lock:
            h5_path = self._sett_misc_outs_dir / self._save_h5_name

            with h5py.File(h5_path, mode='a', driver=None) as h5_hdl:
                self._write_ref_rltzn(h5_hdl)
                self._write_sim_rltzn(h5_hdl)
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

    def _write_sim_rltzn(self, h5_hdl):

        # Should be called by _write_rltzn with a lock

        main_sim_grp_lab = 'data_sim_rltzns'

        sim_grp_lab = self._rs.label

        if not main_sim_grp_lab in h5_hdl:
            sim_grp_main = h5_hdl.create_group(main_sim_grp_lab)

        else:
            sim_grp_main = h5_hdl[main_sim_grp_lab]

        if not sim_grp_lab in sim_grp_main:
            sim_grp = sim_grp_main.create_group(sim_grp_lab)

        else:
            sim_grp = sim_grp_main[sim_grp_lab]

        for var in vars(self._rs):
            var_val = getattr(self._rs, var)
            if isinstance(var_val, np.ndarray):
                sim_grp[var] = var_val

            elif isinstance(var_val, dict):
                for key, value in var_val.items():
                    if isinstance(value, np.ndarray):

                        if isinstance(key, tuple) and len(key) == 2:

                            if all([isinstance(x, str) for x in key]):
                                comb = f'{var}_' + '_'.join(key)

                            else:
                                comb = f'{var}_{key[0]}_{key[1]:03d}'

                        else:
                            raise NotImplementedError(key)

                        sim_grp[comb] = value

                    elif isinstance(value, (int, float, str, tuple)):
                        if var not in sim_grp:
                            sim_grp_sub = sim_grp.create_group(var)

                        else:
                            sim_grp_sub = sim_grp[var]

                        if isinstance(value, (int, float)):
                            sim_grp_sub.attrs[key] = value

                        elif isinstance(value, (str, tuple)):
                            sim_grp_sub.attrs[key] = str(value)

                        else:
                            raise NotImplementedError((key, value))

                    else:
                        raise NotImplementedError((key, value))

            elif isinstance(var_val, (int, float, str, tuple)):

                if isinstance(var_val, (int, float)):
                    sim_grp.attrs[var] = var_val

                elif isinstance(var_val, (str, tuple)):
                    sim_grp.attrs[var] = str(var_val)

                else:
                    raise NotImplementedError((var, var_val))

            elif isinstance(var_val, type(None)):
                print(f'{var} is None!')
                continue

            else:
                raise NotImplementedError((var, var_val))

        h5_hdl.flush()
        return
