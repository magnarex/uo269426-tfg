import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
from DQM.classes.metrics.Metric import Metric
from DQM.classes.metrics.MSE import MSE
from DQM.classes.filters.Filter import Filter
from DQM.classes.filters.MinMax import MinMax

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data
from DQM.classes.Model import Model


begin_log(parent,'main')



train, valid = Data('A','eta').training_validation(Fgood=0.6,Fbad=0.8)
model = Model()
model.train(train,2)
model.add_metric(MSE,'test_mse')
model.add_filter(
    MinMax,
    metric_alias='test_mse',
    args=(0,1.5e-6)
)