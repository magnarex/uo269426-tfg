#!/usr/bin/env python
# coding: utf-8

# # Paquete de utilidades para los DataFrames de Pandas

# ## Dependencias

# In[9]:


import numpy as np
import pandas as pd
from IPython.display import HTML, display


# ## Código

# ### Variables de utilidad

# In[ ]:





# ### Funciones estéticas

# In[10]:


def df_pprint(df,head=3, hide_index=False):
    df1 = df.head(3).style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    df1.set_properties(**{'text-align': 'center', 'float_format':'{:,.0f}'.format})
    if hide_index: df1.hide_index()
    display(HTML(df1.to_html()))


# In[11]:


def pretty_table(df,head=3, hide_index=False):
    df1 = df.head(3).style.set_table_styles([dict(selector='th', props=[('text-align', 'center')])])
    df1.set_properties(**{'text-align': 'center', 'float_format':'{:,.0f}'.format})
    if hide_index: df1.hide_index()
    display(HTML(df1.style.to_html()))


# In[51]:


def pretty_table(df,head=3, hide_index=False):
    if hide_index: df.hide_index()
    from pandas.io.formats.style import Styler
    data_props = 'color: white;; text-align:center;font-size:12pt'
    index_props = 'color:white;text-align:center;font-size:14pt'
    my_css = {
        "row_heading": "",
        "col_heading": "",
        "index_name": "",
        "col": "c",
        "row": "r",
        "col_trim": "",
        "row_trim": "",
        "level": "l",
        "data": "",
        "blank": "",
    }
    html = Styler(df, uuid_len=0, cell_ids=False)
    html.set_table_styles([{'selector': 'td', 'props': data_props},
                        {'selector': '.l0', 'props': index_props}],
                        css_class_names=my_css)
    display(HTML(html.to_html()))


# ### Funciones útiles

# In[12]:


def str2arr(string):
    return np.array(string[1:-1].split(',')).astype(np.float64)


# In[13]:


def hist2arr(df):
    pass


# In[14]:


def read_file(obvs,period='C'):
    pass
    


# ## Exportar a .py

# In[15]:


if __name__ == '__main__':
    get_ipython().system('jupyter nbconvert --to python df_utils.ipynb')

