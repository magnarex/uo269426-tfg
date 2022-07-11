import sys
import os

parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import pickle
from sklearn.decomposition import NMF
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import threading
from time import perf_counter


from DQM.utils.data import parent, train_cfg_dict, obvs_list, filter_cfg_dict
from DQM.utils.logging import begin_log
from DQM.utils.threading import ReturningThread

from DQM.classes.filters.MinMax import MinMax
from DQM.classes.Model import Model
from DQM.classes.HyperData import HyperData
from DQM.classes.metrics.Metric import Metric
from DQM.classes.metrics.MSE import MSE





class HyperModel(object):
    def __init__(self) -> None:
        self.models = {}
    
    def train_parallel(self,train_hdata : HyperData):
        threads = []
        models = []
        for obvs in obvs_list:
            t = ReturningThread(
                target = self.train_model,
                kwargs=dict(obvs=obvs,train_set = train_hdata.sets[obvs]),
                name=f'training : {obvs}'
            )
            t.start()
            threads.append(t)
        
        for thread in threads:
            models.append(thread.join())
        
        # for model

    def train_serial(self,train_hdata : HyperData):
        for obvs in obvs_list:
            logging.debug(f'Entrenando el modelo de {obvs}')
            self.train_model(**dict(obvs=obvs,train_set = train_hdata.sets[obvs]))
          

    def train(self, train_hdata,mode='serial'):
        if mode == 'serial':
            time_i = perf_counter()
            self.train_serial(train_hdata)
            time_f = perf_counter()
            logging.info(f'El tiempo que ha llevado entrenar a todos los modelos en serie es {time_f-time_i}')
        elif mode == 'parallel':
            time_i = perf_counter()
            self.train_parallel(train_hdata)
            time_f = perf_counter()
            logging.info(f'El tiempo que ha llevado entrenar a todos los modelos en paralelo es {time_f-time_i}')
        else:
            raise ValueError(f'El valor {mode} no está definido como modo. Por favor, indique uno de entre "serial" y "parallel".')


    def train_model(self,train_set,obvs):
        model = Model()
        model.train(train_set,**train_cfg_dict[obvs])
        self.models[obvs] = model
        return model
    

    def add_metric(self, metric : Metric, alias : str = None):
        logging.debug('Añadiendo métricas...')
        for model in self.models.values():
            model : Model
            obvs = model.obvs
            logging.debug(f'Añadiendo métrica para {obvs}...')
            model.add_metric(metric, alias+f'_{obvs}')


    def add_filter(self,obvs,alias,metric_alias,filter_cfg):
        filters_dict = {
            'MinMax'    :   MinMax
        }
        filter_cfg = filter_cfg.copy()
        filter_cfg['filter'] = filters_dict[filter_cfg['filter']]
        self.models[obvs].add_filter(
            alias=alias+f'_{obvs}',
            metric_alias = metric_alias+f'_{obvs}',
            **filter_cfg
        )


    def add_filters(self,alias,metric_alias,filter_cfg = None):
        logging.debug('Añadiendo filtros...')
        if filter_cfg is None: filter_cfg = filter_cfg_dict.copy()

        for obvs in obvs_list:
            logging.debug(f'Añadiendo filtro para {obvs}')
            self.add_filter(obvs,alias,metric_alias,filter_cfg[obvs])


    def rmv_filters(self,alias,metric_alias):
        logging.debug('Eliminando filtros...')
        for obvs in obvs_list:
            logging.debug(f'Eliminando filtro para {obvs}')
            self.models[obvs].rmv_filter(alias+f'_{obvs}',metric_alias+f'_{obvs}')

    def save(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent + '/hyper/models'

        if filename is None:
            filename = f'HyperModel'
        
        pickle.dump(self,open(f'{parentdir}/{filename}.hmodel','wb+'))
        logging.info(f'El modelo ha sido guardado en "{parentdir}/{filename}.hmodel".')

    def load(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent +'/hyper/models'
        if filename is None:
            filename = f'HyperModel'

        model = pickle.load(open(f'{parentdir}/{filename}.hmodel','rb'))
        
        logging.info(f'El modelo ha sido cargado de "{parentdir}/{filename}.hmodel".')
        return model

    
    def update(self,other):
        self.models = other.models
        return self

    def eval_model(self,valid_data,obvs):
        model : Model = self.models[obvs]
        return model.eval(valid_data)
    
    def eval_metrics(self,valid_data,obvs):
        model : Model = self.models[obvs]
        model.eval_metrics(valid_data)


    def eval(self,valid_hdata,filter_cfg=None):
        if filter_cfg is None: filter_cfg = filter_cfg_dict.copy()
        self.real_labels = pd.DataFrame({
            obvs : valid_hdata.sets[obvs].data['labels']
            for obvs in obvs_list
        })
        threads = {}
        labels = {}
        for obvs in obvs_list:
            t = threading.Thread(
                target = self.eval_metrics,
                kwargs=dict(
                    obvs=obvs,
                    valid_data = valid_hdata.sets[obvs]
                ),
                name=f'eval : {obvs}'
            )
            t.start()
            threads[obvs] = t
        
        for thread in threads.values():
            thread.join()

        for obvs in obvs_list:
            metric = self.models[obvs].metrics[f'MSE_{obvs}'].metric
            labels[obvs] = (metric <= filter_cfg[obvs]['args'][1])
            
        self.reco_labels = pd.DataFrame(labels)
    
    def confusion(self,valid_hdata : HyperData):
        self.eval(valid_hdata)

        real = np.prod(self.real_labels.values,axis=1)
        recon = np.prod(self.reco_labels.values,axis=1)
        total = len(recon)

        for obvs in obvs_list:
            print('\n\n')
            logging.info(f'Matriz de confusión de {obvs}')
            self.stats(self.real_labels[obvs].values,self.reco_labels[obvs].values)
        print('\n\n')
        logging.info(f'Matriz de confusión conjunta')
        return self.stats(real,recon)
    

    def stats(self,real,recon):
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
        # MCC = (TP*TN-FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
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
            # f'Matthews Correlation Coefficient (MCC):\t{MCC:.3f}\n'
            f'Fowlkes-Mallows Index (FM):\t{FM:.3f}\n'
            f'Informedness (BM):\t{BM:.3f}\n'
            f'Markedness (MK):\t{MK:.3f}\n'
            f'Diagnostic odds ratio (DOR):\t{DOR:.3f}\n'

        )

        stats = {
            'TP'    :TP,
            'TN'    :TN,
            'FP'    :FP,
            'FN'    :FN,
            'TPR'   :TPR,
            'TNR'   :TNR,
            'PPV'   :PPV,
            'FOR'   :FOR,
            'ACC'   :ACC
        }

        return stats


    
    def plot_metrics(self,alias):
        for obvs in obvs_list:
            alias_i = alias + f'_{obvs}'
            self.models[obvs].plot_metric(
                alias=alias_i,
                doCut = True,
                doShow = True,
                cut = filter_cfg_dict[obvs]['args'][1]
            )
        

if __name__ == '__main__':
    begin_log(parent,'HyperModel')
    # htrain = HyperData().load('HTrain')
    hvalid = HyperData().load('HValid')
    hmodel : HyperModel = HyperModel().load()
    hmodel.add_metric(MSE,'MSE')
    hmodel.add_filters('MinMax','MSE')
    hmodel.confusion(hvalid)
    hmodel.plot_metrics('MSE')



