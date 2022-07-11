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

class Entries(Filter):
    __name__ = 'Entries'
    def __init__(self,target,min_entries):
        self.min_entries = min_entries
        super().__init__(target, min_entries)
    
    def filter(self,target):
        # print('Self Min',self.min,type(self.min))
        # print('Target Metric',target.metric,type(target.metric))
        return target.data['entries'] >= self.min_entries

    def __str__(self):
        return f'#entries en {self.target} >= {self.min_entries}'



if __name__ == '__main__':
    from DQM.classes.Data import Data
    begin_log(parent,'Entries')