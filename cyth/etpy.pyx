# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np


cdef extern from "math.h" nogil:
    cdef:
        DT_D log(DT_D x)


cpdef void fill_bin_idxs_ts(
        const DT_D[::1] probs,
              DT_UL[::1] bins_ts, 
        const DT_UL n_bins) except +:

    cdef:
        Py_ssize_t i, n_vals = probs.shape[0]

    assert bins_ts.shape[0] == n_vals
    assert n_vals >= n_bins

    for i in range(n_vals):    
        bins_ts[i] = 0

    for i in range(n_vals):    
        bins_ts[i] = <DT_UL> (n_bins * probs[i])

    return


cpdef void fill_bin_dens_1d(
        const DT_UL[::1] bins_ts,
              DT_D[::1] bins_dens) except +:
    
    cdef:
        Py_ssize_t i
        DT_UL n_bins = bins_dens.shape[0]

    for i in range(n_bins):
        bins_dens[i] = 0

    # Frequencies.
    for i in range(bins_ts.shape[0]):
        bins_dens[bins_ts[i]] += 1

    # Densities.
    for i in range(n_bins):
        bins_dens[i] /= n_bins

    return


cpdef void fill_bin_dens_2d(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
              DT_D[:, ::1] bins_dens_xy) except +:

    cdef:
        Py_ssize_t i, j
        DT_UL n_vals = bins_ts_x.shape[0], n_bins = bins_dens_xy.shape[0]
        DT_UL n_bins_sq = n_bins**2

    assert bins_dens_xy.shape[1] == n_bins
    assert n_vals == bins_ts_y.shape[0]

    for i in range(n_bins):
        for j in range(n_bins):
            bins_dens_xy[i, j] = 0

    # Frequencies.
    for i in range(n_vals):
        bins_dens_xy[bins_ts_x[i], bins_ts_y[i]] += 1

    # Densities.
    for i in range(n_bins):
        for j in range(n_bins):
            bins_dens_xy[i, j] /= n_bins_sq
    return


cpdef void fill_etpy_lcl_ts(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
        const DT_D[::1] bins_dens_x,
        const DT_D[::1] bins_dens_y,
              DT_D[::1] etpy_ts,
        const DT_D[:, ::1] bins_dens_xy) except +:

    cdef:
        Py_ssize_t i
        DT_D dens, prod
        DT_UL n_vals = bins_ts_x.shape[0], n_bins = bins_dens_xy.shape[0]

    assert bins_dens_x.shape[0] == n_bins
    assert bins_dens_y.shape[0] == n_bins
    assert bins_dens_xy.shape[1] == n_bins
    assert n_vals == bins_ts_y.shape[0]
    assert n_vals == etpy_ts.shape[0]

    for i in range(n_vals):    
        etpy_ts[i] = 0

    for i in range(n_vals):
        dens = bins_dens_xy[bins_ts_x[i], bins_ts_y[i]]

        if not dens:
            continue
        
        prod = bins_dens_x[bins_ts_x[i]] * bins_dens_y[bins_ts_y[i]]

        etpy_ts[i] = -(dens * log(dens / prod))

    return