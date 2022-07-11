import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

from DQM.utils.data import parent
from DQM.utils.logging import begin_log

from DQM.classes.Data import Data
from DQM.classes.Model import Model
from DQM.classes.metrics.MSE import MSE
from DQM.classes.filters.MinMax import MinMax

begin_log(parent,'filter_demo')

eval_set = Data().load('eta')
model = Model().load(filename='eta_LS-all')
test_mse = model.add_metric(MSE,'MSE')
model.add_filter(
        MinMax,
        metric_alias='MSE',
        args=(0,4e-6)
    )
model.confusion(eval_set)