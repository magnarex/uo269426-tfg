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

from DQM.utils.data import parent
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data


begin_log(parent,'graphs')

xlabel = {
    'eta'   :   r'Pseudorrapidez, $\eta$ (uds. arb.)',
    'phi'   :   r'Ángulo azimutal, $\varphi$ (rad)',
    'pt'    :   r'Momento transverso, $p_T$ (GeV/c)',
    'chi2'  :   r'Coeficiente ji cuadrado, $\chi^2$ (uds. arb.)',
}
obvs_list = ['eta','phi','pt','chi2']
obvs_list = ['pt']
for obvs in obvs_list:

    data_list = [Data(period = period_i,obvs=obvs) for period_i in ['A','B','C','D']]
    data = sum(data_list,start=Data(load=False))

    bins = data.bins
    edges = data.edges
    counts = data.get_all().sum(axis=0)

    fig, ax = plt.subplots(1,1,figsize=(16/2,9/2))
    ax.stairs(counts,edges,fill=True)
    if obvs == 'pt':
        ax.set_xlim(0,100)
    elif obvs == 'chi2':
        ax.set_xlim(0,10)
    else:
        ax.set_xlim(edges.min(),edges.max())
        
    ax.set_xlabel(xlabel[obvs],fontsize=12)
    ax.set_ylabel('Número de sucesos (uds. arb.)',fontsize=12)
    ax.grid(linestyle='--')
    plt.savefig(parent+f'/graphs/dist/{obvs}.jpg',dpi=300)

