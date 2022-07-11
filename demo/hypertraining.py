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
import copy
import matplotlib.pylab as pl
from matplotlib.colors import ListedColormap

from DQM.classes.metrics.MSE import MSE
from DQM.classes.filters.MinMax import MinMax

from DQM.utils.data import parent, filter_cfg_dict, obvs_list
from DQM.utils.logging import begin_log
from DQM.classes.Data import Data
from DQM.classes.Model import Model
from DQM.classes.HyperModel import HyperModel
from DQM.classes.HyperData import HyperData

def eval_model(obvs,eval_data:HyperData,hmodel : HyperModel,thresh):

    obvs_i,obvs_j = obvs
    thresh_ran_i = np.logspace(*thresh[0])
    thresh_ran_j = np.logspace(*thresh[1])

    thresh_mesh_i,thresh_mesh_j = np.meshgrid(thresh_ran_i,thresh_ran_j)
    thresh_mesh_i = thresh_mesh_i.ravel()
    thresh_mesh_j = thresh_mesh_j.ravel()
    thresh = list(zip(thresh_mesh_i,thresh_mesh_j))
    index = pd.MultiIndex.from_tuples(thresh, names=[obvs_i,obvs_j])


    stats_list = []
    for k,(thresh_i,thresh_j) in enumerate(thresh):

        filter_cfg_i = copy.deepcopy(filter_cfg_dict)
        filter_cfg_i[obvs_i]['args'][1] = thresh_i
        filter_cfg_i[obvs_j]['args'][1] = thresh_j

        hmodel.add_filters('MinMax','MSE',filter_cfg=filter_cfg_i)
        stats_list.append(hmodel.confusion(eval_data))
        hmodel.rmv_filters('MinMax','MSE')

    df = pd.DataFrame(stats_list,index=index,columns=['TP','TN','FP','FN','TPR','TNR','ACC','FOR'])
    # logging.info(df)
    return df



def gen_model_stats(eval_data,obvs,hmodel,thresh,overwrite=False):
    
    

    stats = eval_model(obvs,eval_data,hmodel,thresh=thresh)


    with open(parent+f'/hyper/eval/{"_".join(obvs)}.df','wb+') as file:
        pickle.dump(stats,file)


title_style = {'fontsize':14}
fig_params = {'size' : (20,15),'constrained_layout':True}

def plot_stats(obvs_i,obvs_j,doShow=False):

    with open(parent+f'/hyper/eval/{obvs_i}_{obvs_j}.df','rb') as file:
        df = pickle.load(file)
    
    stats = df.copy()
    N = int(np.sqrt(stats.index.values.size))
    x = np.array([index[0] for index in stats.index.values])
    x = x.reshape(N,N)

    y = np.array([index[1] for index in stats.index.values])
    y = y.reshape(N,N)


    symbol_dict = {
        'eta'   :   r'{\eta}',
        'phi'   :   r'{\varphi}',
        'pt'    :   r'{p_T}',
        'chi2'  :   r'{\chi^2}'
    }

    eps_dict = {
        'eta'   :   r'{\eta}',
        'phi'   :   r'{\varphi}',
        'pt'    :   r'{p_T}',
        'chi2'  :   r'{\chi^2}'
    }



    xticks = np.logspace(
        np.log10(x.min()),
        np.log10(x.max()),
        int(np.log10(x.max()))-int(np.log10(x.min()))+1
        )

    yticks = np.logspace(
        np.log10(y.min()),
        np.log10(y.max()),
        int(np.log10(y.max()))-int(np.log10(x.min()))+1
        )


    fig,ax = plt.subplots(1,2,figsize=(16/3,16/3))

    cmap_dict = {
        'TNR'   :   pl.cm.Blues,
        'ACC'   :   pl.cm.Oranges,
        'FOR'   :   pl.cm.Greens,
    }

    for i,col in enumerate(['TNR','ACC','FOR']):
        z = stats[col].values
        z = z.reshape(N,N)
        cmap = cmap_dict[col]
        # cmap = cmap_(np.arange(cmap_.N))
        # cmap[:,-1] = np.linspace(0,1,cmap_.N)
        # cmap = ListedColormap(cmap)

        ax[0].plot(x.ravel(),z.ravel(),label=col,marker='x',linestyle='none')
        ax[0].set_xlabel(obvs_i)
        ax[1].plot(y.ravel(),z.ravel(),label=col,marker='x',linestyle='none')
        ax[1].set_xlabel(obvs_j)
        # cmap = pl.cm.viridis
        # artist = ax.contourf(
        #     x,
        #     y,
        #     z,
        #     label=col,
        #     levels = 20,
        #     cmap = cmap,
        #     alpha = 0.3
        # )

        # ax.set_xlabel(r'Valor del parámetro de corte del MSE de ${}$, $\varepsilon_{}$ (u.a.)'.format(symbol_dict[obvs_i],symbol_dict[obvs_i]))
        # ax.set_ylabel(r'Valor del parámetro de corte del MSE de ${}$, $\varepsilon_{}$ (u.a.)'.format(symbol_dict[obvs_j],symbol_dict[obvs_j]))

        # ax.set_title('Entrenadas y evaluadas en LS',**title_style)
        # fig.colorbar(artist,ax=ax,location='right')


    for ax in fig.get_axes():
        ax.legend()
        # ax.set_xscale('log')
        # ax.set_yscale('log')
    #     ax.legend(loc=4)
    #     # ax.set_ylim(-0.01,1.01)
    #     ax.set_xlim(x.min(),x.max())
    #     ax.set_aspect('auto')
    #     ax.set_xticks(xticks)
    #     ax.set_yticks(np.linspace(0,1,6))
    #     ax.tick_params(
    #         axis = 'x',
    #         which = 'minor',
    #         labelbottom=False
    #     )
    #     ax.grid()
    #     ax.grid(which='minor',axis='x',linestyle='--',linewidth=0.3)

    plt.tight_layout()
    # fig.savefig(parent+f'/graphs/stats_hyper/stats_{obvs_i}_{obvs_j}.jpeg',dpi=300)
    if doShow: plt.show()
    plt.close()
    del fig




def main():
    thresh_dict = {
        'eta'   :   (np.log10(4.5e-6),np.log10(6.5e-6),51),
        'phi'   :   (np.log10(4e-6),np.log10(6e-6),51),
        'pt'    :   (np.log10(8.5e-7),np.log10(10.5e-7),51),
        'chi2'  :   (np.log10(4.5e-6),np.log10(6.5e-6),51)
    }
    hvalid = HyperData().load('HValid')
    hmodel : HyperModel = HyperModel().load()
    hmodel.add_metric(MSE,'MSE')

    pairs = []


    for obvs_i in obvs_list:
        for obvs_j in obvs_list:
            if obvs_i == obvs_j: continue
            if [obvs_i,obvs_j] in pairs or [obvs_j,obvs_i] in pairs: continue
            plot_stats(obvs_i,obvs_j, doShow=True)
            # gen_model_stats(
            #     hmodel = hmodel,
            #     eval_data=hvalid,
            #     obvs=[obvs_i,obvs_j],
            #     thresh=[thresh_dict[obvs_i],thresh_dict[obvs_j]]
            # )
            # plot_stats(obvs_i,obvs_j)
            pairs.append([obvs_i,obvs_j])

if __name__ == '__main__':
    begin_log(parent,'training')
    main()
        