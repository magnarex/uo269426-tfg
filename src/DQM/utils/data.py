#!/usr/bin/env python
# coding: utf-8

# # Paquete de utilidades para los datos

# ## Dependencias

# In[6]:


from pathlib import Path
import os
import json

# ## Código

# ### Nombre de los archivos

# In[7]:




# ### Directorio padre

# In[8]:


parent = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'

with open(parent+'/meta/file_names.json','r+') as file:
    names = json.load(file)

with open(parent+'/meta/train_data.json','r+') as file:
    train_data_dict = json.load(file)

with open(parent+'/meta/train_cfg.json','r+') as file:
    train_cfg_dict = json.load(file)

with open(parent+'/meta/filter_cfg.json','r+') as file:
    filter_cfg_dict = json.load(file)

with open(parent+'/meta/cut_cfg.json','r+') as file:
    cut_cfg_dict = json.load(file)

with open(parent+'/meta/info.json','r+') as file:
    info = json.load(file)
    obvs_list = info['obvs']
    era_list = info['eras']