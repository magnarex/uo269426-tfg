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
        Esta función sirve para iniciar un objeto Filter. Guarda los parámetros necesarios como atributos.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.
            - Fgood     :   Fracción de los datos buenos que formará parte del conjunto de entrenamiento.
            - Fbad      :   Fracción de los datos malos que formará parte del conjunto de entrenamiento.
        '''

        self.Fgood = Fgood
        self.Fbad = Fbad
        logging.debug('Creamos un filtro a la clase Data para crear el set de entrenamiento'
        ' y el de validación.')

        super().__init__(target, Fgood, Fbad)
    
    def filter(self, target):
        """
        Esta función es la que calcula el filtro.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        """


        data = target.data
        good_train = self.divide_set(target,True,int(self.Fgood*target.Ngood))
        bad_train = self.divide_set(target,False,int(self.Fbad*target.Nbad))

        mask = np.isin(data.index.values, bad_train + good_train)
        N = len(good_train)+len(bad_train)
        Ngood = data['labels'].values[mask].sum()
        Fgood = Ngood/N
        logging.debug(
            'Hemos creado un filtro Training con las siguientes características:\n'
            f'\t- total:\t{N:.0f}\n'
            f'\t- good:\t{Ngood:.0f} ({Fgood*100:2.2f}%)\n'
            f'\t- bad:\t{N-Ngood:.0f} ({100-Fgood*100:2.2f}%)\n'
        )
        del data
        return mask



    def divide_set(self,target,label:bool,k:int):
        """
        Función auxiliar que nos ayuda a calcular el filtro.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.
            - label     :   Indica si se quieren los datos malos o los buenos.
            - k         :   Número de LS que se quieren tomar del total.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        """
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