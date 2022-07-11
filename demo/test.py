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
import pickle

from DQM.classes.metrics.Metric import Metric
from DQM.classes.metrics.MSE import MSE
from DQM.classes.filters.Filter import Filter
from DQM.classes.filters.MinMax import MinMax

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data
from DQM.classes.Model import Model


begin_log(parent,'test')


A = Data('A','eta').minimum_entries(200).normalize()
B = Data('B','eta').minimum_entries(200).normalize()
C = Data('C','eta').minimum_entries(200).normalize()
D = Data('D','eta').minimum_entries(200).normalize()

df = pd.concat([A.data,B.data,C.data,D.data])
bad = df[df['labels'] == False]
cut = int(len(bad.index)*0.6)

train_df = bad.iloc[:cut,:]
valid_df = bad.iloc[cut:,:]




train = Data('bad','eta',False).load_df(train_df)
valid = Data('bad','eta',False).load_df(valid_df)

model = Model()
model.train(train,3)
model.add_metric(MSE,'test_mse')
model.plot_components()

model.add_filter(
    MinMax,
    metric_alias='test_mse',
    args=(0,5.7e-6)
)
model.roc_curve(valid)


