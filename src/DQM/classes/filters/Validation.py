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
from DQM.classes.filters.Training import Training

from DQM.utils.data import parent
from DQM.utils.logging import begin_log



class Validation(Filter):
    def __init__(self, target):
        super().__init__(target)

    def filter(self, target):
        """
        Esta función es la que calcula el filtro. Intenta adquirir las etiquetas del entrenamiento
         e invertirlas y, si no las hay, da error.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        """

        try:
            mask = np.logical_not(target.training.mask)
        except AttributeError as err:
            logging.info('Primero se tiene que crear el set de entrenamiento.')
            raise err
        return mask

    
