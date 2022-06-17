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




data = Data('D','eta').minimum_entries(200).normalize()
train, valid = data.training_validation(Fgood=0.6,Fbad=0.8)
# valid = Data('C','eta').minimum_entries(200).normalize()
model = Model()
model.train(train,2)
model.add_metric(MSE,'test_mse')
model.add_filter(
    MinMax,
    metric_alias='test_mse',
    args=(0,7e-6)
)
model.eval(valid)
model.confusion(valid)
model.metrics['test_mse'].plot_metric()




# data = Data('D','eta').minimum_entries(200).normalize()
# train, valid = data.training_validation(Fgood=0.6,Fbad=0.8)

# model.save('test_D')


thresh_ran = np.logspace(-6,-5,500+1)
df = pd.DataFrame({},index=thresh_ran,columns=['TP','TN','FP','FN','TPR','TNR','PPV','FOR'])


model = Model.load('test_D')
model.add_metric(MSE,'test_mse')
valid = Data('C','eta').minimum_entries(200).normalize()

for thresh in thresh_ran:
    model.add_filter(
        MinMax,
        metric_alias='test_mse',
        args=(0,thresh)
    )
    df.loc[thresh] = pd.Series(model.confusion(valid))
    model.rmv_filter('MinMax_01','test_mse')

fig,ax = plt.subplots(1,1)
for col in ['TPR','TNR','PPV','FOR']:
    ax.plot(thresh_ran,df[col],label=col)
ax.set_xscale('log')
ax.legend()
plt.show(block=True)