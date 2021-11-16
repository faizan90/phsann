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

        self._dur_tmr_cumm_call_times = {}
        self._dur_tmr_cumm_n_calls = {}

        self._dur_tmr_keep_keys = (
            '_set_lag_nth_wts',
            '_set_label_wts',
            '_set_auto_obj_wts',
            '_search_init_temp',
            )

        self._reset_timers()
        return

    def _reset_timers(self):

        '''
        NOTE: Timers are reset automatically in _simu_grp. _gen_gnrc_rltzn
        also updates it manually.

        Methods that are called before the simulation starts such as,
        Automatic temperature search and objective function weights,
        if saved, remain in the dictionaries but the rest are deleted
        each time a new simulation starts. The methods whose info you
        want to keep have to be added to self._dur_tmr_keep_keys manually.
        '''

        # Putting keys into a list is important, due to the dynamic size change
        # problem of the dictionary.
        for key in list(self._dur_tmr_cumm_call_times.keys()):
            if key in self._dur_tmr_keep_keys:
                continue

            else:
                del self._dur_tmr_cumm_call_times[key]
                del self._dur_tmr_cumm_n_calls[key]

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

            if meth_name not in self._dur_tmr_cumm_call_times:
                self._dur_tmr_cumm_call_times[meth_name] = 0.0
                self._dur_tmr_cumm_n_calls[meth_name] = 0

            self._dur_tmr_cumm_call_times[meth_name] += (end - beg)
            self._dur_tmr_cumm_n_calls[meth_name] += 1

            return res

        return wrap

    _timer_wrap = staticmethod(_timer_wrap)

