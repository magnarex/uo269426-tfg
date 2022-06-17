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
from DQM.classes.filters.Filter import Filter
from DQM.utils.data import parent
from DQM.utils.logging import begin_log



class Training(Filter):
    def __init__(self, target, Fgood=0.6, Fbad=0.8):
        '''
        Params:
            - Fgood :   Porcentaje de los datos buenos. Por defecto: 60%.
            - Fbad  :   Porcentaje de los datos malos. Por defecto: 80%.
        '''

        self.Fgood = Fgood
        self.Fbad = Fbad
        logging.info('Creamos un filtro a la clase Data para crear el set de entrenamiento'
        ' y el de validación.')

        super().__init__(target, Fgood, Fbad)
    
    def filter(self, target):
        """
        No queremos guardar target en memoria porque suelen ser un montón de datos ~1e5.
        """
        data = target.data
        good_train = self.divide_set(target,True,int(self.Fgood*target.Ngood))
        bad_train = self.divide_set(target,False,int(self.Fbad*target.Nbad))

        train = np.isin(data.index.values, bad_train + good_train)
        N = len(good_train)+len(bad_train)
        Ngood = data['labels'].values[train].sum()
        Fgood = Ngood/N
        logging.info(
            'Hemos creado un filtro Training con las siguientes características:\n'
            f'\t- total:\t{N:.0f}\n'
            f'\t- good:\t{Ngood:.0f} ({Fgood*100:2.2f}%)\n'
            f'\t- bad:\t{N-Ngood:.0f} ({100-Fgood*100:2.2f}%)\n'
        )
        return train


    def divide_set(self,target,label:bool,k:int):
        df = target.data
        index = df[df['labels']==label].index.to_list()
        # logging.debug('Index:\n',type(index),index,'\n')
        train = random.sample(index,k = k)
        return train


if __name__ == '__main__':
    begin_log(parent,'Training')
    from DQM.classes.Data import Data
    data = Data('A','eta')
    train, valid = data.training_validation()