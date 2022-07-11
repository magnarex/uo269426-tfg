import sys
import os

parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.decomposition import NMF
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import concurrent.futures
import threading

from DQM.utils.data import parent, train_cfg_dict, obvs_list, era_list
from DQM.utils.threading import ReturningThread
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data


save_threads = []
for obvs in obvs_list:
    t_result = []
    threads = []
    for era in era_list:
        t = ReturningThread(target = Data,kwargs=dict(period=era,obvs=obvs),name=f'{obvs}_{era}')
        t.start()
        threads.append(t)

    for thread in threads:
        t_result.append(thread.join())
    
    data_i = sum(t_result,start=Data(obvs=obvs,load=False))
    data_i.minimum_entries(200).normalize()

    t = threading.Thread(target=data_i.save,kwargs=dict(filename=f'{obvs}'),name=f'save_{obvs}')
    t.start()
    save_threads.append(t)

for thread in save_threads:
        thread.join()