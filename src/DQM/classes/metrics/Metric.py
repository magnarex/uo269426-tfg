import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import numpy as np
import matplotlib.pyplot as plt
import logging

from DQM.utils.data import parent
from DQM.utils.logging import begin_log





class Metric(object):
    __name__ = 'Metric'
    filters = {}
    def __init__(self,model,alias):
        logging.info('Inicializando clase basada en la clase Metric...')
        self.model = model
        self.alias = alias
    
    def eval(self,data):
        self.data_labels = data.data['labels']
        self.data_bins = data.bins.copy()
        self.matrix = matrix = self.metric_func(data)
        self.metric = matrix.mean(axis=1)


    def metric_func(self,data):
        # Do stuff with the data in the model. Return metric values for each row.
        logging.info('Métrica de prueba.')
        return np.ones(self.src.Nentries)

    def plot_metric(self):
        fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
        src = self.model.src
        data = src.data
        metric_obvs = self.matrix.mean(axis=0)
        metric_entries = self.metric
        labels = data['labels']
        bad = data.index[labels == False]

        ax1.set_title('Distribución sobre el observable')
        ax1.step(
            src.bins,
            metric_obvs,
            where='mid',
            label=self.__name__,
            linewidth = 1.5,
            color = 'xkcd:royal blue',
        )

        ax1.set_yscale('log')
        ax1.legend()

        ax2.set_title('Distribución sobre la métrica')
        ax2.hist(
            metric_entries,
            label=self.__name__,
            linewidth = 1.5,
            color = 'xkcd:royal blue',
            histtype='step',
            log=True,
            bins = 100
        )
        ax2.legend()

        ax3.set_title('Distribución sobre las entradas')
        ax3.step(
            range(src.Nentries),
            metric_entries,
            where='mid',
            label=self.__name__,
            linewidth = 1.5,
            color = 'xkcd:royal blue'
        )
        # ax3.hlines(bad,metric_entries.min(),metric_entries.max())
        ax3.set_yscale('log')

        ylim3 = ax3.get_ylim()
        ax3.vlines(
            bad,
            ylim3[0],
            ylim3[1],
            linewidth=0.5,
            color = 'r',
            linestyle = 'dashed'
        )
        
        
        ax3.legend()

        plt.show(block=True)


if __name__ == '__main__':
    begin_log(parent,'Metric')