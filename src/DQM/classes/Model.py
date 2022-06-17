import sys
import os

from sklearn import metrics
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
from DQM.classes.filters.Filter import Filter



class Model(object):
    #TODO: Hay que arreglar la forma en la que el modelo almacena los datos con los que se
    # ha entrenado, ya que hace que almacenarlo sea muy costoso.

    #TODO: Añadir funcionalidad de entrenar y evaluar por separado.
    seed = None
    N = None
    tol = None

    flags = {
        'trained'   : False,
    }

    metrics = {}

    def __init__(self):
        logging.info('Se ha creado el objeto Modelo a partir del objeto Data.')

    def train(self,train_set,N,max_iter=10000,tol=1e-4,seed=None):     
        self.period = train_set.period
        self.obvs = train_set.obvs
        self.bins = train_set.bins.copy()
        V = train_set.get_all()


        if seed is None: seed = np.random.randint(1,1e4)
        self.seed = seed

        self.N = N
        self.max_iter = max_iter
        self.tol = tol

        model = NMF(
            N,
            max_iter=max_iter,
            init='nndsvda',
            tol=tol,
            random_state=seed,
        )

        logging.info(f'Comienza el entrenamiento del modelo con {N} componentes y {max_iter} iteraciones máximas.')
        self.W = model.fit(V)
        logging.info('El entrenamiento del modelo ha finalizado.')

        # self.components = model.components_
        self.model = model
        self.flags['trained'] = True

        del V
        del train_set

    def recon(self,test_set):
        W = self.model.transform(test_set.get_all())
        return np.dot(W,self.model.components_)



    def save(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent + '/models'

        if filename is None:
            filename = f'N{self.N}_S{self.seed}_{self.src.period}_{self.src.obvs}'
        
        self.model_info()
        pickle.dump(self,open(f'{parentdir}/{filename}.model','wb+'))
        logging.info(f'El modelo ha sido guardado en "{parentdir}/{filename}.model".')

    def load(filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent +'/models'

        model = pickle.load(open(f'{parentdir}/{filename}.model','rb'))
        
        logging.info(f'El modelo ha sido cargado de "{parentdir}/{filename}.model".')
        model.model_info()
        return model
    
    def model_info(self):
        logging.info(
            'Información del modelo:\n'
            f'\tN:\t\t{self.N}\n'
            f'\tmax_iter:\t{self.max_iter}\n'
            f'\ttol:\t\t{self.tol}\n'
            f'\tseed:\t\t{self.seed}\n'
            f'\ttrained:\t{self.flags["trained"]}\n'
            f'\tData:\n'
            f'\t\tperiod:\t{self.period}\n'
            f'\t\tobvs:\t{self.obvs}\n'
            '\tmetrics:\n'
            +
            '\n'.join([f'\t\t- {metric_alias}\t: {metric.__name__}' for metric_alias, metric in self.metrics.items()])
            +
            '\n'
            +
            '\tfilters:\n'
            +
            '\n'.join(sum([
                    [
                        f'\t\t- {filter_alias} ({filter.__name__})\t: {str(filter)}'
                        for filter_alias,filter in metric.filters.items()
                    ]
                    for metric_alias,metric in self.metrics.items()
                ],
                []    
            ))
            +
            '\n'
        )


    def plot_components(self):
        fig,ax = plt.subplots(1,1)
        comp = self.components
        ax.step(self.bins,comp.T,where='mid')
        ax.legend([f'Comp. {i}' for i in range(self.N)])
        plt.show(block=True)

    
    def add_metric(self,metric,alias=None):
        metric_name = metric.__name__
        keys = self.metrics.keys()
        if alias is None:
            metrics = [key.split('_')[0] for key in keys]
            Nfits = sum([1 if key == metric_name else 0 for key in metrics])
            alias = f'{metric_name}_{str(Nfits+1).zfill(2):2}'
        elif alias in keys:
            raise ValueError('El nombre ya existe, por favor, utilice otro.')

        metric_obj = metric(self,alias=alias)
        self.metrics[alias] = metric_obj

        logging.info(
            f'Se ha añadido la métrica {metric_name} bajo el alias {alias}.'
        )
        self.model_info()

    def plot_entry(self,entry):
        data = self.src
        comp = self.components
        if not 0 <= entry < data.Nentries:
            raise ValueError('El valor introducido de "entry" no está en el rango de entradas.') 
        
        real = data.data['histo'][entry]
        w = self.W[entry]
        recon = np.dot(w,comp)
        bins = self.src.bins

        
        fig,ax = plt.subplots(1,1)
        ax.step(
            bins,
            real,
            where='mid',
            label='original',
            linewidth = 1.5,
            color = 'xkcd:royal blue')

        for i in range(self.N):
            ax.step(
                bins,
                w[i]*comp[i],
                where='mid',
                label=f'Comp. {i}',
                linewidth = 1.3,
                linestyle='dashed')
        ax.step(
            bins,
            recon,
            where='mid',
            label='Recon',
            linestyle = 'dashed',
            color = 'k',
            linewidth = 2)
        ax.legend(loc=1)

        metric_text = '\n'.join(
            [
                f'{metric.alias} ({metric.__name__}):\t{metric.metric[entry]:.2e}'
                for metric in self.metrics.values()
            ]
        ).expandtabs()
        
        props = dict(
            boxstyle='round',
            facecolor='wheat',
            alpha=0.5
        )
        
        ax.text(
            0.01,
            0.95,
            metric_text,
            transform=ax.transAxes,
            fontsize=14,
            verticalalignment='top',
            bbox=props
        )

        plt.show(block=True)





    def plot_metric(self,alias):
        self.metrics[alias].plot_metric()




    def add_filter(self, filter : Filter, metric_alias, args = [], alias=None,
                   min=-np.inf,max=np.inf):

        if metric_alias not in self.metrics:
            raise ValueError(f'No existe ninguna métrica con el alias {metric_alias}')
        metric : Metric = self.metrics[metric_alias]
        metric_name = metric.__name__
        filter_name = filter.__name__
        

        keys = metric.filters.keys()
        
        if alias is None:
            filters = [key.split('_')[0] for key in keys]
            Nfits = sum([1 if key == filter_name else 0 for key in filters])
            alias = f'{filter_name}_{str(Nfits+1).zfill(2):2}'
        elif alias in keys:
            raise ValueError('El nombre ya existe, por favor, utilice otro.')
        
        filter_obj = filter(metric,*args)
        logging.info(f'Filtrando {alias} ({filter_name}): {str(filter_obj)}.')
        
        
        metric.filters[alias] = filter_obj
        self.model_info()
    






    def rmv_filter(self,alias,metric_alias):
        if metric_alias not in self.metrics:
            raise ValueError(f'No existe ninguna métrica con el alias {metric_alias}')
        metric : Metric = self.metrics[metric_alias]

        if alias not in metric.filters:
            raise ValueError(f'No existe ninguna métrica con el alias {alias}')

        filter = metric.filters[alias]
        name = filter.__name__
        logging.info(f'Eliminando el filtro {alias} ({name}): {filter.min} < {metric.__name__} < {filter.max}.')
        del metric.filters[alias]
        self.model_info()




    def eval(self):

        #TODO: Evaluar la medición del modelo según los datos dados.
        #TODO: Dar una lista de las métricas/filtros que se quieren evaluar.
        # Reconstruye los datos introducidos
        # Calcula las métricas
        # Colapsa todos los filtros para calcular el valor de las etiquetas.
        labels = np.ones(self.src.Nentries)
        for filter in self.filters.values():
            labels *= filter.value
        self.recon_labels = labels
    
    def confusion(self,test_set):

        real = test_set.data['labels']
        recon = self.recon_labels
        total = test_set.Nentries
        
        is_P = recon == True
        is_N = recon == False
        P = (real==True).sum()
        N = (real==False).sum()
        PP = is_P.sum()
        NN = is_N.sum()


        TP = np.where(recon[is_P]==real[is_P])[0].size
        TN = np.where(recon[is_N]==real[is_N])[0].size
        FP = PP-TP
        FN = NN-TN
        
        TPR = TP/P
        TNR = TN/N
        PPV = TP/(TP+FP)
        NPV = TN/(TN+FN)
        FDR = FP/(FP+TP)
        FOR = FN/(TN+FN)
        FNR = FN/P
        FPR = FP/N

        LRp = TPR/FPR
        LRn = FNR/TNR
        PT = np.sqrt(FPR)/(np.sqrt(TPR)+np.sqrt(FPR))
        TS = TP/(TP+FN+FP)

        Prev = P/(P+N)
        ACC = (TP+TN)/(P+N)

        BA = (TPR+TNR)/2
        F1 = 2*TP/(2*TP+FP+FN)
        MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
        FM = np.sqrt(PPV*TPR)
        BM = TPR+TNR-1
        MK = PPV+NPV-1
        DOR = LRp/LRn

        logging.info(
            f'\nMatriz de confusión:\n'
            f' \t  PP  \t  NN  \n'
            f'P\t{TP:.0f}\t{FN:.0f}\n'
            f'N\t{FP:.0f}\t{TN:.0f}\n\n'

            f'Sensitivity (TPR):\t{TPR*100:2.2f}%\n'
            f'Specifity (TNR):\t{TNR*100:2.2f}%\n'
            f'Precision (PPV):\t{PPV*100:2.2f}%\n'
            f'False Omission Rate (FOR):\t{FOR*100:2.2f}%\n'

            f'Prevalence Thershold (PT):\t{PT*100:2.2f}%\n'
            f'Prevalence:\t{Prev*100:2.2f}%\n'

            f'Accuracy (ACC):\t{ACC*100:2.2f}%\n'
            f'F1 score:\t{F1:1.3f}\n'

            f'Threat Score (TS):\t{TS:.3f}\n'
            f'Balanced Accuracy (BA):\t{BA:.3f}\n'
            f'Matthews Correlation Coefficient (MCC):\t{MCC:.3f}\n'
            f'Fowlkes-Mallows Index (FM):\t{FM:.3f}\n'
            f'Informedness (BM):\t{BM:.3f}\n'
            f'Markedness (MK):\t{MK:.3f}\n'
            f'Diagnostic odds ratio (DOR):\t{DOR:.3f}\n'

        )





    
        



if __name__ == '__main__':
    begin_log(parent,'Model')
    from DQM.classes.metrics.MSE import MSE
    from DQM.classes.metrics.Metric import Metric
    from DQM.classes.Data import Data
    # data = Data('A','eta').clean()
    # model = Model(src=data)
    # model.train(6)
    # model.save('test')
    model = Model.load(filename='test')
    model.plot_components()
    test_mse = model.add_metric(MSE,'test_mse')
    model.plot_metric('test_mse')
    model.plot_entry(5234)
    model.plot_entry(5235)
    model.plot_entry(5236)


