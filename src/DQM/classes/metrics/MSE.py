import sys
import os

from scipy.fft import skip_backend
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import threading

from DQM.utils.data import parent,obvs_list, era_list
from DQM.utils.data import parent,obvs_list, era_list
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
    from DQM.classes.Data import Data

    skip_list = []
    # obvs_list = ['eta']
    skip_list = [
        ['pt','A'],
        ['pt','B'],
        ['pt','C'],
        'eta',
        'phi',
        # 'pt',
        'chi2'
    ]
    for obvs in obvs_list:
        if obvs in skip_list: continue
        model = Model.load(f'{obvs}_LS-all')
        for era in era_list:
            if [obvs,era] in skip_list: continue

            valid = Data(era,obvs).minimum_entries(1).normalize()
            model.add_metric(MSE,f'MSE_{era}')
            logging.info(
                'Estudio de los datos:\n'
                f'\t- Observable:\t{obvs}\n'
                f'\t- Era:\t\t{era}\n'
            )
            model.eval_metrics(valid,metrics=[f'MSE_{era}'])
            model.plot_metric(f'MSE_{era}',doShow=False,doCut=True)