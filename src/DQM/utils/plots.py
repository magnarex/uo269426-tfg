#!/usr/bin/env python
# coding: utf-8

# # Paquete de utilidades para los DataFrames de Pandas

# ## Dependencias

# In[17]:


import matplotlib.pyplot as plt
import numpy as np
import os


# ## Código

# ### Funciones de representación

# In[18]:


# def plot_arr2hist(arr):
#     pass


def plot_df2hist(df,row,path='graphs/test/',name='',save=True):
    histo = df.at[row,'histo']/df.at[row,'entries']
    bins=np.linspace(df.at[row,'Xmin'],df.at[row,'Xmax'],df.at[row,'Xbins'])

    fig, ax = plt.subplots(1,1)

    _ = ax.step(bins,histo,where='mid',linewidth=0.3)
    if save:
        try:
            plt.savefig(path+'/'+str(row).zfill(3)+'.png')
        except FileNotFoundError:
            os.makedirs(path)
            plt.savefig(path+'/'+str(row).zfill(3)+'.png')
        plt.close()


# ## Exportar a .py

# In[19]:


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to python plot_utils.ipynb')

