import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.filters.Filter import Filter
from DQM.classes.metrics.Metric import Metric

class MinMax(Filter):
    __name__ = 'MinMax'
    def __init__(self,target:Metric,min,max):
        '''
        Esta función sirve para iniciar un objeto Filter. Guarda los parámetros necesarios como atributos.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.
            - min       :   Cota mínima del valor.
            - max       :   Cota máxima del valor.
        '''
        self.min = min
        self.max = max
        super().__init__(target, min, max,eval=False)
    
    def filter(self,target:Metric):
        '''
        Esta función es la que calcula el filtro.

        NOTA: Se sospecha que esta función puede tener un bug, ya que hay valores que no parecen fiables.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        '''
        return (self.min <= target.metric)&(target.metric <= self.max)

    def __str__(self):
        '''
        Sobrecargamos el método "__str__" para poder hacer un logging más efectivo.
        '''
        return f'{self.min} <= {self.target} <= {self.max}'