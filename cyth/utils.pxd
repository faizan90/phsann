# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True


cpdef void adjust_phss_range(
        const double[:, :] phss, double[:, :] phss_adj) except +