'''
Created on Dec 27, 2019

@author: Faizan
'''

from multiprocessing import current_process

from .core.main import PhaseAnnealing

from .core.plot import PhaseAnnealingPlot

current_process().authkey = 'phsann'.encode(encoding='utf_8', errors='strict')
