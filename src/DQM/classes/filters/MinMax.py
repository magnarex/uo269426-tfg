import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.filters.Filter import Filter
from DQM.classes.metrics.Metric import Metric

class MinMax(Filter):
    __name__ = 'MinMax'
    def __init__(self,target:Metric,min,max):
        self.min = min
        self.max = max
        super().__init__(target, min, max,doFilter=False)
    
    def filter(self,target:Metric):
        return (self.min <= target.metric)&(target.metric <= max)

    def __str__(self):
        return f'{self.min} <= {self.target} <= {self.max}'



if __name__ == '__main__':
    begin_log(parent,'MinMax')