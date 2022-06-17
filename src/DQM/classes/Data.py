import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import copy

from DQM.utils.data import names, parent
from DQM.utils.dataframes import str2arr
from DQM.utils.logging import begin_log
from DQM.classes.filters.Filter import Filter
from DQM.classes.filters.Training import Training
from DQM.classes.filters.Validation import Validation







class Data(object):
    __name__ = 'Data'
    flags = {
        'nonzero' : False,
        'normalized' : False
    }

    def __init__(self,period=None,obvs=None,load = True):
        if isinstance(period,type(None)) and isinstance(obvs,type(None)) or not load:
            self.period = period
            self.obvs = obvs
            logging.info('Se ha creado satisfactoriamente el objeto Data.')
        elif not isinstance(period,type(None)) and not isinstance(obvs,type(None)):
            self.load(period,obvs)
        else:
            raise AttributeError('Debes indicar ambos argumentos o ninguno. Por favor, indica los valores del periodo y el observable de estudio.')

            


    def load(self,period,obvs):
        data = pd.read_csv(f'{parent}/data/csv/'+names[obvs][period]+'.csv')
        data['histo'] = data['histo'].apply(str2arr)
        logging.info(f'Se han leído los datos del fichero {names[obvs][period]}.csv satisfactoriamente.')

        self.data = data
        self.period = period
        self.obvs = obvs
        self.update_attributes()

        del data


    def load_df(self,df):
        logging.info('Se han leído los datos desde un objeto pandas.DataFrame.')
        self.data = df
        self.update_attributes()
    
    def normalize(self):
        # Ahora las cuentas de los histogramas están en tantos por uno.
        data = self.data.copy()
        self.data['histo'] = data['histo']/data['entries']
        self.flags['normalized'] = True

        del data

        logging.info('Los valores de los histogramas han sido normalizados y ahora se encuentran en tanto por uno.')
        return self

    def nonzero(self):
        # Eliminamos las entradas vacías
        data = self.data.copy()
        nonzero = data.entries != 0
        data = data[nonzero]
        self.data = data
        
        self.Nzeros = np.logical_not(nonzero).sum()
        self.Fzeros = self.Nzeros/self.Nentries
        logging.info(
            f'Se han eliminado {self.Nzeros} entradas vacías,'
            f'que se corresponden con un {self.Fzeros*100:2.2f}% del total y '
            f'un {self.Nzeros/self.Nbad*100:2.2f}% de las entradas malas.\n'
        )
        
        
        self.update_attributes()
        self.flags['nonzero'] = True

        del data
        del nonzero

        return self

    def training(self,Fgood=0.6,Fbad=0.8):
        self.training = train = Training(self,Fgood,Fbad)
        data = self.data
        train_idx = data.index.values[train.mask]
        logging.info('Se procede a crear el training Data:')
        trainData = Data(self.period,self.obvs,False)
        trainData.load_df(data.loc[train_idx,:])



        return trainData

    def validation(self):
        self.validation = valid = Validation(self)
        data = self.data
        valid_idx = data.index.values[valid.mask]
        
        logging.info('Se procede a crear el validation Data:')
        validData = Data(self.period,self.obvs,False)
        validData.load_df(data.loc[valid_idx,:])
        return validData
    
    def training_validation(self,Fgood=0.6,Fbad=0.8):
       
        trainData = self.training(Fgood,Fbad)
        validData = self.validation()

        return trainData,validData



    
    def get_bad(self,col='histo'):
        data = self.data
        bad = data[data['labels'] == False]
        return np.stack(bad[col].to_numpy())

    def get_good(self,col='histo'):
        data = self.data
        good = data[data['label'] == True]
        return np.stack(good[col].to_numpy())

    def get_all(self,col='histo'):
        data = self.data
        return np.stack(data[col].to_numpy())


    def clean(self):
        self.nonzero()
        self.normalize()
        return self


    def update_attributes(self):
        data = self.data.copy()

        self.Nentries = len(data.index)
        self.edges = edges = np.linspace(data['Xmin'].values[0],data['Xmax'].values[0],num=data['Xbins'].values[0]+1)
        self.bins = (edges[1:]+edges[:-1])/2

        good = data['labels']==True
        self.Ngood = len(data[good])
        self.Fgood = self.Ngood/self.Nentries

        bad  = data['labels']==False
        self.Nbad = len(data[bad])
        self.Fbad = self.Nbad/self.Nentries

        self.ratio = self.Ngood/self.Nbad

        logging.info(
            'Se han actualizado los valores de los atributos:\n'
            f'\tNúmero de entradas:\t\t{self.Nentries}\n'
            f'\tNúmero de entradas buenas:\t{self.Ngood}\t({self.Fgood*100:2.2f}%)\n'
            f'\tNúmero de entradas malas:\t{self.Nbad}\t({self.Fbad*100:2.2f}%)\n'
            f'\tHay {self.ratio:.0f} entradas buenas por cada una mala.\n'
        )

        del data


    

if __name__ == '__main__':
    begin_log(parent,'Data')
    a = Data('A','eta')
    a.clean()