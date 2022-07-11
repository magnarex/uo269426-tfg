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
import concurrent.futures
import threading


from DQM.utils.data import parent, train_cfg_dict, obvs_list
from DQM.utils.logging import begin_log
from DQM.utils.threading import ReturningThread
from DQM.classes.Data import Data
from DQM.classes.filters.Training import Training
from DQM.classes.filters.Validation import Validation




class HyperData(object):
    def __init__(self):
        self.sets = {}

    def load_sets(self, parentdir=None):
        threads = {}

        for obvs in obvs_list:
            t = ReturningThread(
                target=self.load_set,
                kwargs=dict(filename=obvs,parentdir=parentdir),
                name=f'load_sets : {obvs}'
            )
            t.start()
            threads[obvs] = t

        chi2 = threads['chi2'].join()
        pt   = threads['pt'  ].join()
        eta  = threads['eta' ].join()
        phi  = threads['phi' ].join()

        self.sets = {
            'eta'   :   eta,
            'phi'   :   phi,
            'pt'    :   pt,
            'chi2'  :   chi2,
        }


    
    def load_set(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent +'/data/sets'
        data = pickle.load(open(f'{parentdir}/{filename}.dat','rb'))
        return data
    
    def save(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent + '/hyper'

        if filename is None:
            filename = f'Data'
        
        pickle.dump(self,open(f'{parentdir}/{filename}.hdat','wb+'))
        logging.debug(f'Los datos han sido guardados en "{parentdir}/{filename}.data".')
    
    def load(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent +'/hyper'
        if filename is None:
            filename = 'Data'

        data = pickle.load(open(f'{parentdir}/{filename}.hdat','rb'))
        
        logging.debug(f'Los datos han sido cargados de "{parentdir}/{filename}.model".')
        return data


    def apply_mask(self,mask):
        masked_sets = {}

        for key,value in self.sets.items():
            masked_sets[key] = self.sets[key].apply_mask(mask)
        
        return masked_sets



    def training_validation(self,Fgood=0.6,Fbad=0.8,sample='eta'):
        #Primero con una (sample) y luego con el resto:
        train_mask = Training(self.sets[sample],Fgood,Fbad).mask

        trainHData = HyperData()
        trainHData.sets = self.apply_mask(train_mask)

        validHData = HyperData()
        validHData.sets = self.apply_mask(np.logical_not(train_mask))

        return trainHData, validHData




if __name__ == '__main__':
    begin_log(parent,'HyperData')
    # hdata = HyperData()
    # hdata.load_sets()
    # hdata.save()
    # hdata = hdata.load()
    htrain,hvalid = HyperData().training_validation(Fgood=0.6,Fbad=0.8)
    htrain.save('HTrain')
    hvalid.save('HValid')
    # htrain = HyperData().load('HTrain')
    # hvalid = HyperData().load('HValid')




