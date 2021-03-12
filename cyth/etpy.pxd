# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

ctypedef double DT_D
ctypedef unsigned long DT_UL

cpdef void fill_bin_idxs_ts(
        const DT_D[::1] probs,
              DT_UL[::1] bins_ts, 
        const DT_UL n_bins) except +

cpdef void fill_bin_dens_1d(
        const DT_UL[::1] bins_ts,
              DT_D[::1] bins_dens) except +

cpdef void fill_bin_dens_2d(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
              DT_D[:, ::1] bins_dens_xy) except +

cpdef void fill_etpy_lcl_ts(
        const DT_UL[::1] bins_ts_x,
        const DT_UL[::1] bins_ts_y,
        const DT_D[::1] bins_dens_x,
        const DT_D[::1] bins_dens_y,
              DT_D[::1] etpy_ts,
        const DT_D[:, ::1] bins_dens_xy) except +
