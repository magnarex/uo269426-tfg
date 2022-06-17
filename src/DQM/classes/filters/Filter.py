import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from DQM.classes.metrics.Metric import Metric
from DQM.utils.data import parent
from DQM.utils.logging import begin_log



class Filter(object):
    __name__ = 'Filter'
    def __init__(self,target,*params,eval=True):
        self.target = target.__name__
        self.params = params
        if eval: self.eval(target)
    
    def filter(self,target):
        pass

    def eval(self,target):
        self.mask = self.filter(target)

    def __str__(self):
        return 'Representación del filtro en string.'

if __name__ == '__main__':
    begin_log(parent,'Filter')
    from DQM.classes.metrics.MSE import MSE
    from DQM.classes.metrics.Metric import Metric
    from DQM.classes.Model import Model
    model = Model().load(filename='test')
    test_mse = model.add_metric(MSE,'test_mse')
    filter_ms = model.add_filter('test_mse',0,1e-4)
    model.eval_labels()
    model.confusion()
