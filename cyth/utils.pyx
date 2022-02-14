# cython: nonecheck=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3
# cython: infer_types=False
# cython: embedsignature=True

cdef extern from "math.h" nogil:
    cdef:
        double M_PI

import numpy as np
cimport numpy as np


cpdef void adjust_phss_range(
        const double[:, :] phss, double[:, :] phss_adj) except +:

    cdef:
        Py_ssize_t i, j
        double phs, ratio

    assert phss.shape[0] == phss_adj.shape[0], (
        phss.shape, phss_adj.shape)

    assert phss.shape[1] == phss_adj.shape[1], (
        phss.shape, phss_adj.shape)

    for i in range(phss.shape[0]):
        for j in range(phss.shape[1]):

            phs = phss[i, j]

            if phs > +M_PI:
                ratio = (phs / +M_PI) - 1
                phs = -M_PI * (1 - ratio)

            elif phs < -M_PI:
                ratio = (phs / -M_PI) - 1
                phs = +M_PI * (1 - ratio)

            phss_adj[i, j] = phs

    return