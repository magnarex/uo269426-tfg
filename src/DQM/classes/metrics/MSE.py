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
import matplotlib.pyplot as plt

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.metrics.Metric import Metric



class MSE(Metric):
    __name__ = 'MSE'
    def __init__(self, model,alias):
        super().__init__(model,alias)
        logging.info('Clase MSE basada en la clase Metric inicializada correctamente.')

    def metric_func(self,data):
        return (data.get_all()-self.model.recon(data))**2



if __name__ == '__main__':
    begin_log(parent,'MSE')
    from DQM.classes.Model import Model
    model = Model().load(filename='test')
    model.plot_components()