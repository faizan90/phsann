# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

import numpy as np
cimport numpy as np

cpdef np.ndarray get_phs_corrs(const double[:] phs_spec) except +

cpdef void adjust_phss_range(
        const double[:, :] phss, double[:, :] phss_adj) except +