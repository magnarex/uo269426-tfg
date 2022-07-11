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

class Entries(Filter):
    __name__ = 'Entries'
    def __init__(self,target,min_entries,eval=True):
        '''
        Esta función sirve para iniciar un objeto Filter. Guarda los parámetros necesarios como atributos.

        - Variables:
            - target        :   Objeto sobre el cual se va a aplicar el filtro.
            - min_entries   :   Mínimo de entradas aceptadas.
            - eval          :   Valor booleano que indica si el filtro se evalúa nada más iniciarse
                                o no. Habrá casos en los que no nos interese.
        '''
        self.min_entries = min_entries
        super().__init__(target, min_entries,eval=eval)
    
    def filter(self,target):
        '''
        Esta función es la que calcula el filtro.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        '''
        return target.data['entries'] >= self.min_entries

    def __str__(self):
        '''
        Sobrecargamos el método "__str__" para poder hacer un logging más efectivo.
        '''
        return f'#entries en {self.target} >= {self.min_entries}'
