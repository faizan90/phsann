'''
@author: Faizan

18 Jun 2020

12:30:04

Timer wrapper from stackoverflow
'''

from timeit import default_timer
from functools import wraps


class PhaseAnnealingBase:

    def __init__(self, verbose):

        '''
        Parameters
        ----------
        verbose : bool
            Whether to show activity messages
        '''

        assert isinstance(verbose, bool), 'verbose not a boolean!'

        self._vb = verbose

        self._reset_timers()
        return

    def _reset_timers(self):

        '''
        NOTE: Timers are reset automatically in _sim_grp. _gen_gnrc_rltzn
        also updates it manually.
        '''

        self._sim_tmr_cumm_call_times = {}
        self._sim_tmr_cumm_n_calls = {}
        return

    def _timer_wrap(meth):

        @wraps(meth)
        def wrap(self, *args, **kwargs):
            beg = default_timer()

            res = meth(self, *args, **kwargs)

            end = default_timer()

            meth_name = meth.__name__

            if meth_name == '_gen_gnrc_rltzn':
                raise Exception

            if meth_name not in self._sim_tmr_cumm_call_times:
                self._sim_tmr_cumm_call_times[meth_name] = 0.0
                self._sim_tmr_cumm_n_calls[meth_name] = 0

            self._sim_tmr_cumm_call_times[meth_name] += (end - beg)
            self._sim_tmr_cumm_n_calls[meth_name] += 1

            return res

        return wrap

    _timer_wrap = staticmethod(_timer_wrap)

