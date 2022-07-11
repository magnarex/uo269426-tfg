import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from DQM.classes.metrics.Metric import Metric
from DQM.utils.data import parent
from DQM.utils.logging import begin_log



class Filter(object):
    '''
    Esta clase nos permite producir etiquetas a partir de una condición dada. Se usará para
    producir las etiquetas de la certificación, entre otros.
    '''

    __name__ = 'Filter'


    def __init__(self,target,*params,eval=True):
        '''
        Esta función sirve para iniciar un objeto Filter.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.
            - params    :   Parámetros del filtro, usados por algunas clases basadas en ésta.
            - eval      :   Valor booleano que indica si el filtro se evalúa nada más iniciarse
                            o no. Habrá casos en los que no nos interese.
        '''
        self.target = target.__name__
        self.params = params
        if eval: self.eval(target)
    


    def filter(self,target):
        '''
        Esta función es la que calcula el filtro. Las clases basadas en esta modifican esta función
        para producir diferentes filtrados. En este caso, es sólo un dummy.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.

        - Returns:
            - mask      :   Valores booleanos que constituyen el filtro.
        '''
        pass



    def eval(self,target):
        '''
        Esta función es la que evalúa el filtro. Llama al método "filter"; está programada de esta
        manera para que la función "eval" sea universal para todos los filtros pero la función
        "filter" sea única. Crea el atributo "mask", que contiene los valores booleanos del filtro
        para que se puedan acceder a ellos desde fuera.

        - Variables:
            - target    :   Objeto sobre el cual se va a aplicar el filtro.
        '''
        self.mask = self.filter(target)



    def __str__(self):
        '''
        Sobrecargamos el método "__str__" para poder hacer un logging más efectivo. Para cada clase
        de tipo Filter será diferente. Esta función es sólo un dummy.
        '''
        return 'Representación del filtro en string.'
    
