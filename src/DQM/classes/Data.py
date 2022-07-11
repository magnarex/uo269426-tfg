import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import logging
import copy
import pickle

from DQM.utils.data import names, parent
from DQM.utils.dataframes import str2arr
from DQM.utils.logging import begin_log
from DQM.classes.filters.Filter import Filter
from DQM.classes.filters.Entries import Entries
from DQM.classes.filters.Training import Training
from DQM.classes.filters.Validation import Validation







class Data(object):
    __name__ = 'Data'
    flags = {
        'nonzero' : False,
        'normalized' : False
    }

    def __init__(self,period='',obvs='',load = True):
        if period == '' and obvs == '' or not load:
            self.period = period
            self.obvs = obvs
            self.data = pd.DataFrame({})
            logging.debug('Se ha creado satisfactoriamente el objeto Data.')
        elif not period == '' and not obvs == '':
            self.load_csv(period,obvs)
        else:
            raise AttributeError('Debes indicar ambos argumentos o ninguno. Por favor, indica los valores del periodo y el observable de estudio.')

            


    def load_csv(self,period,obvs):
        data = pd.read_csv(f'{parent}/data/csv/'+names[obvs][period]+'.csv')
        data['histo'] = data['histo'].apply(str2arr)
        logging.debug(f'Se han leído los datos del fichero {names[obvs][period]}.csv satisfactoriamente.')

        self.data = data
        self.period = period
        self.obvs = obvs
        self.update_attributes()

        del data


    def load_df(self,df):
        logging.debug('Se han leído los datos desde un objeto pandas.DataFrame.')
        self.data = df
        self.update_attributes()
        return self


    def save(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent + '/data/sets'

        if filename is None:
            filename = f'{self.obvs}_{self.period}'
        
        pickle.dump(self,open(f'{parentdir}/{filename}.dat','wb+'))
        logging.info(f'Los datos han sido guardados en "{parentdir}/{filename}.dat".')
    

    def load(self,filename=None,parentdir=None):
        if parentdir is None:
            parentdir = parent + '/data/sets'

        data = pickle.load(open(f'{parentdir}/{filename}.dat','rb'))
        
        logging.info(f'Los datos han sido cargados de "{parentdir}/{filename}.dat".')
        return data
    


    def normalize(self):
        # Ahora las cuentas de los histogramas están en tantos por uno.
        data = self.data.copy()
        self.data['histo'] = data['histo']/data['entries']
        self.flags['normalized'] = True

        del data

        logging.debug('Los valores de los histogramas han sido normalizados y ahora se encuentran en tanto por uno.')
        return self

    def minimum_entries(self,min_entries):
        # Eliminamos las entradas que no llegan al mínimo de eventos
        data = self.data.copy()
        filter = Entries(self,min_entries)
        data = data[filter.mask]
        self.data = data
        labels = data['labels'].values
        
        self.Nmin = np.logical_not(filter.mask).sum()
        self.Fmin = self.Nmin/self.Nentries

        Ngood = np.sum(labels)
        Nbad = np.sum(np.logical_not(labels))

        #TODO: Terminar de hacer esto, el mensaje no se muestra bien.


        logging.debug(
            f'Se han eliminado {self.Nmin} entradas, con #entries <= {min_entries}, '
            f'que se corresponden con un {self.Fmin*100:2.2f}% del total. '
            f'Se han eliminado un {(1-Nbad/self.Nbad)*100:2.2f}% de las entradas malas\n'
            
        )
        
        
        self.update_attributes()
        self.flags['nonzero'] = True

        del data

        return self

    def apply_mask(self,mask):
        data = self.data
        idx = data.index.values[mask]
        maskedData = Data(self.period,self.obvs,False)
        maskedData.load_df(data.loc[idx,:])

        return maskedData

    def training(self,Fgood=0.6,Fbad=0.8):
        self.training = train = Training(self,Fgood,Fbad)
        return self.apply_mask(train.mask)



    def validation(self):
        return self.apply_mask(np.logical_not(self.training.mask))
    


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

        logging.debug(
            'Se han actualizado los valores de los atributos:\n'
            f'\tNúmero de entradas:\t\t{self.Nentries}\n'
            f'\tNúmero de entradas buenas:\t{self.Ngood}\t({self.Fgood*100:2.2f}%)\n'
            f'\tNúmero de entradas malas:\t{self.Nbad}\t({self.Fbad*100:2.2f}%)\n'
            f'\tHay {self.ratio:.0f} entradas buenas por cada una mala.\n'
        )

        del data

    def collapse_LS(self):
        data = self.data
        hname  = data.loc[0]['hname']
        Xbins  = data.loc[0]['Xbins']
        Xmin  = data.loc[0]['Xmin']
        Xmax  = data.loc[0]['Xmax']
        new_data = pd.DataFrame(columns=data.columns).drop('fromlumi',axis=1)
        runs = set(data['fromrun'].values)

        logging.debug('Se procede a colapsar las LS:')
        for fromrun in runs:
            slice = data[data['fromrun'] == fromrun]
            entries = np.sum(slice['entries'])
            histo = np.stack(slice['histo'].to_numpy()).sum(axis=0)
            labels = slice['labels'].values[0] #porque es la misma para todas las LS de una run
            new_row = {
                'fromrun'   : fromrun,
                'labels'    : labels,
                'hname'     : hname,
                'histo'     : [histo],
                'entries'   : entries,
                'Xbins'     : Xbins,
                'Xmin'      : Xmin,
                'Xmax'      : Xmax
            }

            new_data = pd.concat([new_data,pd.DataFrame(new_row)],ignore_index=True,axis=0)
        logging.debug('Se ha terminado de colapsar las LS.\n')
        self.data = new_data
        self.update_attributes()
        # logging.info(f'{new_row}')

        return self

    def __add__(self,other):
        df_self = self.data.copy()
        df_other = other.data.copy()
        try:
            assert (self.obvs == other.obvs) | (self.obvs == '') | (other.obvs == '')
        except AssertionError:
            raise ValueError('No se pueden sumar objetos Data con distinto observable.')

        return Data(obvs=self.obvs,period=self.period+other.period,load=False).load_df(pd.concat([df_self,df_other]))



    

if __name__ == '__main__':
    begin_log(parent,'Data')

    logging.debug('A')
    a = Data('A','eta')
    a.minimum_entries(200)

    logging.debug('B')
    b = Data('B','eta')
    b.minimum_entries(200)

    logging.debug('C')
    c = Data('C','eta')
    c.minimum_entries(200)

    logging.debug('D')
    d = Data('D','eta')
    d.minimum_entries(200)

    ab = sum([a,b],start=Data(load=False))