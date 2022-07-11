from shutil import which
import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import pandas as pd
import numpy as np
import random
import json
import logging
import matplotlib.pyplot as plt
import pickle

from DQM.classes.metrics.MSE import MSE
from DQM.classes.filters.MinMax import MinMax

from DQM.utils.data import parent, train_cfg_dict, train_data_dict, obvs_list
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data
from DQM.classes.Model import Model







def data_same(obvs,period1,mode1,period2,mode2):

    data = Data(period1,obvs).minimum_entries(200)
    if mode1 == 'Run':
        data = data.collapse_LS()
    data = data.normalize()

    return data.training_validation(Fgood=0.6,Fbad=0.8)


def data_diff(obvs,period1,mode1,period2,mode2):
    train = Data(period1,obvs).minimum_entries(200)
    if mode1 == 'Run':
        train = train.collapse_LS()
    train = train.normalize()

    eval = Data(period2,obvs).minimum_entries(200)
    if mode2 == 'Run':
        eval = eval.collapse_LS()
    eval = eval.normalize()

    return train,eval

func_dict = {
    'data_same'    : data_same,
    'data_diff'    : data_diff
}

def data(obvs,data_dict):
    data_func = func_dict[data_dict['func']]
    train_data, eval_data = data_func(obvs,**data_dict['kwargs'])
    return train_data, eval_data


def train_model(train_data,train_cfg={}):
    model = Model()
    model.train(train_data,**train_cfg)
    model.add_metric(MSE,'test_mse')
    
    return model

def eval_model(eval_data:Data,model:Model,thresh):

    thresh_ran = np.logspace(*thresh)
    df = pd.DataFrame({},index=thresh_ran,columns=['TP','TN','FP','FN','TPR','TNR','ACC','FOR'])
    
    
    for thresh in thresh_ran:
        model.add_filter(
            MinMax,
            metric_alias='MSE',
            args=(0,thresh)
        )
        df.loc[thresh] = pd.Series(model.confusion(eval_data))
        model.rmv_filter('MinMax_01','MSE')

    return df

def train_eval(obvs,data_dict,thresh,train_cfg={}):
    train_data,eval_data = data(obvs,data_dict)
    model = train_model(train_data,train_cfg=train_cfg)
    stats = eval_model(eval_data,model,thresh)
    return model,stats




def gen_dict():
    training_dict = {}
    periods = ['A','B','C','D']
    # modes = ['Run','LS']
    modes = ['LS']
    
    args_per = []
    for period_1 in periods:
        args_per += [[period_1,period_2] for period_2 in periods]

    args_mode = []
    for mode_1 in modes:
        args_mode += [[mode_1,mode_2] for mode_2 in modes]

    for per_i in args_per:
        per1,per2 = per_i
        if per1 == per2: func = 'data_same'
        else: func = 'data_diff'

        for mode_j in args_mode:
            mode1,mode2 = mode_j
            if per1 == per2 and mode1 != mode2: continue
            id = f'{mode1}-{per1}_{mode2}-{per2}'
            dict_ij = {
                'name'      :   f'{id}',
                'func'      :   func,
                'kwargs'    :   {
                    'period1'   :   per1,
                    'mode1'     :   mode1,
                    'period2'   :   per2,
                    'mode2'     :   mode2,
                }
            }
            training_dict[id] = dict_ij


    with open(parent+'/meta/train_data.json','w+') as file:
        json.dump(training_dict, file,indent=4)


def gen_model_stats(obvs,train_cfg,thresh,overwrite=False):
    model_path = ''
    doOW = overwrite

    for key,data_dict in train_data_dict.items():
        name = data_dict['name']
        model_name = name.split('_')[0]
        if parent+f'/models/{obvs}_{model_name}.model' != model_path:
            doOW = overwrite


        model_path = parent+f'/models/{obvs}_{model_name}.model'
        train_data,eval_data = data(obvs,data_dict=data_dict)

        if os.path.isfile(model_path) and not overwrite:
            with open(model_path,'rb') as file:
                model = pickle.load(file)

        else:
            model = train_model(train_data,train_cfg=train_cfg)
            with open(model_path,'wb+') as file:
                pickle.dump(model,file)
            doOW = False

        try:
            del model.metrics['test_mse']
        except:
            pass
        
        model.add_metric(MSE,'MSE')

        stats = eval_model(eval_data,model,thresh=thresh)


        with open(parent+f'/eval/{obvs}_{name}.df','wb+') as file:
            pickle.dump(stats,file)


title_style = {'fontsize':14}
fig_params = {'size' : (20,15),'constrained_layout':True}

def plot_stats(obvs,per1,per2,doShow=False):

    modes = [['LS','LS']]
    stats = {}

    for mode1,mode2 in modes:
        if per1 == per2 and mode1 != mode2: continue
        id = mode1[0]+mode2[0]
        with open(parent+f'/eval/{obvs}_{mode1}-{per1}_{mode2}-{per2}.df','rb') as file:
            df = pickle.load(file)
        stats[id] = df.copy()

    x = df.index.values

    fig,ax = plt.subplots(1,1,figsize=(16/3,3))
    for col in ['TNR','ACC','FOR']:
        ax.plot(x,stats['LL'][col],label=col)
        # ax.set_title('Entrenadas y evaluadas en LS',**title_style)


    for ax in fig.get_axes():
        ax.set_xscale('log')
        ax.legend(loc=4)
        # ax.set_ylim(-0.01,1.01)
        ax.set_xlim(x.min(),x.max())
        ax.set_aspect('auto')
        ax.set_xticks([1e-7,1e-6,1e-5,1e-4])
        ax.set_yticks(np.linspace(0,1,6))
        ax.tick_params(
            axis = 'x',
            which = 'minor',
            labelbottom=False
        )
        ax.grid()
        ax.grid(which='minor',axis='x',linestyle='--',linewidth=0.3)
        ax.set_xlabel(r'Valor del parámetro de corte del MSE, $\varepsilon$ (u.a.)')
        ax.set_ylabel(r'Valor del índice estadístico (u.a.)')

    plt.tight_layout()
    fig.savefig(parent+f'/graphs/stats_{obvs}/stats_{per1}_{per2}.jpeg',dpi=300)
    if doShow: plt.show()
    plt.close()
    del fig




def gen_plot_stats(obvs):
    periods = ['A','B','C','D']
    
    for period1 in periods:
        for period2 in periods:
            plot_stats(obvs,period1,period2)



def main():
    # gen_dict()
    thresh = (-7,-4,301) #en escala log
    skip_list=[]
    # obvs_list = ['eta']
    skip_list = ['eta']
    for obvs in obvs_list:
        if obvs in skip_list: continue
        train_data,valid_data = Data().load(filename=obvs).training_validation()
        # # model = train_model(train_data,train_cfg=train_cfg_dict[obvs])
        # # with open(parent+f'/models/{obvs}_LS-all.model','wb+') as file:
        # #             pickle.dump(model,file)
        with open(parent+f'/models/{obvs}_LS-all.model','rb') as file:
                    model = pickle.load(file)

        try:
            del model.metrics['test_mse']
        except:
            pass
        
        model.add_metric(MSE,'MSE')

        stats = eval_model(valid_data,model,thresh=thresh)
        # with open(parent+f'/eval/{obvs}_LS-all_LS-all.df','wb+') as file:
        #         pickle.dump(stats,file)
        plot_stats(obvs,'all','all',doShow=False)
    

        # gen_model_stats(obvs,train_cfg_dict[obvs],thresh=thresh)
        gen_plot_stats(obvs)
    
    # plot_stats('eta','A','B',doShow=True)

if __name__ == '__main__':
    begin_log(parent,'training')
    main()
        