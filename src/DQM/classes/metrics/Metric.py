import sys
import os
parentdir = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'
sys.path.insert(0, parentdir+'/src')
del parentdir

import numpy as np
import matplotlib.pyplot as plt
import logging

from DQM.utils.data import parent,cut_cfg_dict
from DQM.utils.logging import begin_log





class Metric(object):
    __name__ = 'Metric'
    filters = {}
    metric = None
    def __init__(self,model,alias):
        logging.info('Inicializando clase basada en la clase Metric...')
        self.model = model
        self.alias = alias
    
    def eval(self,data):
        from DQM.classes.Data import Data
        new_df = data.data.sort_values(by=['fromrun','fromlumi']).copy()
        # new_df = data.data.copy(data)
        new_data = Data(obvs = data.obvs, period = data.period,load=False).load_df(new_df)

        self.data_LS = new_data.data['fromlumi'].values
        self.data_run = new_data.data['fromrun'].values
        self.data_labels = new_data.data['labels'].values
        self.xrange = np.arange(len(self.data_labels))
        self.data_bins = new_data.bins.copy()
        self.data_edges = new_data.edges.copy()
        self.matrix = matrix = self.metric_func(data)
        self.new_matrix = self.metric_func(new_data)
        self.metric = matrix.mean(axis=1)
        self.period = data.period
        self.obvs = data.obvs

        self.data_name = [
            f'Run{self.data_run[i]}\nLS{self.data_LS[i]}'
            for i in range(len(self.data_LS))
        ]


    def metric_func(self,data):
        # Do stuff with the data in the model. Return metric values for each row.
        logging.info('Métrica de prueba.')
        return np.ones(self.src.Nentries)

    def plot_metric(self,doShow = True,doSave = True,doCut=True,cut = None):
        
        label_style = dict(
            fontsize = 16
        )

        title_style = dict(
            fontsize = 16
        )

        obvs = self.model.obvs
        era = self.period
        if cut is None: cut = cut_cfg_dict[obvs][era]

        


        obvs_sym = {
            'eta'   :   r'$\eta$',
            'phi'   :   r'$\varphi$',
            'pt'    :   r'$p_T$',
            'chi2'  :   r'$\chi^2$'
        }

        obvs_name = {
            'eta'   :   'Pseudorrapidez',
            'phi'   :   'Ángulo azimutal',
            'pt'    :   'Momento transverso',
            'chi2'  :   'Coeficiente chi cuadrado'
        }

        obvs_uds = {
            'eta'   :   'uds. arb.',
            'phi'   :   'rad',
            'pt'    :   'GeV/c',
            'chi2'  :   'uds. arb.'
        }


        fig,axs = plt.subplots(2,2,figsize=(16,9))
        
        # axs = ((ax1,ax2),(ax3,ax4))
        gs = axs[1,0].get_gridspec()
        for ax in axs[1,:]:
            ax.remove()
        ax1,ax2 = axs[0]
        ax3 = fig.add_subplot(gs[1,:])
        metric_obvs = self.new_matrix.mean(axis=0)
        metric_entries = self.new_matrix.mean(axis=1)
        N = metric_entries.size

        NGood = (metric_entries <= cut).sum()


        xrange = np.array(self.xrange)
        xticks = np.linspace(0,xrange.max(),16).astype(int)
        # logging.info(f'np.log10({metric_entries.min()}),np.log10({metric_entries.max()}),')
        mse_space = np.logspace(
            np.log10(metric_entries[metric_entries!=0].min()),
            np.log10(metric_entries.max()),
            101
        )
        # logging.info(mse_space)

        ax2_xticks = mse_space[::20]
        bad = np.where(self.data_labels==False)[0]

        # ax1.set_title('Distribución sobre el observable',**title_style)
        ax1.step(
            self.data_bins,
            metric_obvs,
            where='mid',
            label=self.__name__,
            linewidth = 1.5,
            color = 'xkcd:royal blue',
            zorder=7
        )
        ax1.set_ylabel('Error de la reconstrucción,\nMSE (uds.arb.)',**label_style)
        ax1.set_xlabel(f'{obvs_name[obvs]}, {obvs_sym[obvs]} ({obvs_uds[obvs]})',**label_style)
        ax1.set_yscale('log')

        if doCut:
            ax1.axhline(
                cut,
                color='k',
                linestyle='--'
            )
            xmin,xmax = ax1.get_xlim()
            ymin,ymax = ax1.get_ylim()
            r_rectangle = plt.Rectangle(
                (xmin,cut),
                width = xmax-xmin,
                height = ymax-cut,
                fc='xkcd:brick red',
                ec="none",
                alpha=0.7,
                zorder=5)
            ax1.add_patch(r_rectangle)

            g_rectangle = plt.Rectangle(
                (xmin,ymin),
                width = xmax-xmin,
                height = cut-ymin,
                fc='xkcd:grass green',
                ec="none",
                alpha=0.7,
                zorder=4)
            ax1.add_patch(g_rectangle)
            
        # ax1.legend()

        # ax2.set_title('Distribución sobre la métrica',**title_style)

        n,bins,_=ax2.hist(
            metric_entries,
            label=self.__name__,
            linewidth = 1.5,
            color = 'xkcd:royal blue',
            histtype='step',
            log=True,
            bins = mse_space,
            zorder = 7
        )
        # ax2.set_xticks(ax2_xticks)
        ax2.set_xscale('log')
        ax2.set_xlabel('Error de la reconstrucción, MSE (uds. arb.)',**label_style)
        ax2.set_ylabel('Número de cuentas, (uds. arb.)',**label_style)
        
        # logging.info(n)
        # logging.info(bins)
        if doCut:
            ax2.axvline(
                cut,
                color='k',
                linestyle='--'
            )
            xmin,xmax = ax2.get_xlim()
            ymin,ymax = ax2.get_ylim()
            r_rectangle = plt.Rectangle(
                (cut,ymin),
                width = xmax-cut,
                height = ymax-ymin,
                fc='xkcd:brick red',
                ec="none",
                alpha=0.7,
                zorder=5)
            ax2.add_patch(r_rectangle)

            g_rectangle = plt.Rectangle(
                (xmin,ymin),
                width = cut-xmin,
                height = ymax-ymin,
                fc='xkcd:grass green',
                ec="none",
                alpha=0.7,
                zorder=4)
            ax2.add_patch(g_rectangle)
            


        # ax2.legend()
        # ax2.set_xscale('log')

        # ax3.set_title('Distribución sobre las entradas',**title_style)
        ax3.plot(
            self.xrange,
            metric_entries,
            # where='mid',
            # label=self.__name__,
            # linestyle = 'none',
            linewidth=1,
            # marker = 'x',
            color = 'xkcd:royal blue',
            zorder=7
        )
        # ax3.hlines(bad,metric_entries.min(),metric_entries.max())
        ax3.set_yscale('log')

        ylim3 = ax3.get_ylim()
        ax3.vlines(
            bad,
            ylim3[0],
            ylim3[1],
            linewidth=0.5,
            linestyle = 'dashed',
            zorder=6,
            alpha = 0.7,
            color = 'xkcd:mustard'
        )
        ax3.set_xlim(xrange.min(),xrange.max())
        ax3.set_ylabel('Error de la reconstrucción,\nMSE (uds. arb.)',**label_style)
        ax3.set_xlabel('Periodo de toma de datos, run/LS',**label_style)
        ax3.set_xticks(xticks)
        labels = [
            self.data_name[xticks[i]]
            if i % 2 == 0
            else '\n'+self.data_name[xticks[i]]
            for i in range(len(xticks))
        ]
        ax3.set_xticklabels(labels=labels,rotation = 0)

        if doCut:
            ax3.axhline(
                cut,
                color='k',
                linestyle='--',
                zorder=1
            )
            ymin,ymax = ax3.get_ylim()
            r_rectangle = plt.Rectangle(
                (0,cut),
                width = self.xrange.max(),
                height = ymax-cut,
                fc='xkcd:brick red',
                ec="none",
                alpha=0.7,
                zorder=5)
            ax3.add_patch(r_rectangle)

            g_rectangle = plt.Rectangle(
                (0,0),
                width = self.xrange.max(),
                height = cut,
                fc='xkcd:grass green',
                ec="none",
                alpha=0.7,
                zorder=4,
                label = f'NGood: {NGood} / {N}'
            )
            ax3.add_patch(g_rectangle)
            

        ax3.legend(loc=1)
        # plt.tight_layout(pad = 0,h_pad=-1,w_pad=0)


        if doShow: plt.show()
        if doSave: fig.savefig(parent+f'/graphs/mse/mse_{obvs}_LS-{self.model.period}_LS-{era}.jpeg',dpi=300)
        del fig



if __name__ == '__main__':
    begin_log(parent,'Metric')
