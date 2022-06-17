#!/usr/bin/env python
# coding: utf-8

# # Paquete de utilidades para los datos

# ## Dependencias

# In[6]:


from pathlib import Path
import os


# ## Código

# ### Nombre de los archivos

# In[7]:


names = {
    "chi2": {
        "A": "GlbMuon_Glb_chi2OverDf_MuonCert_Labeled_UL2018A_Reduced",
        "B": "GlbMuon_Glb_chi2OverDf_MuonCert_Labeled_UL2018B_Reduced",
        "C": "GlbMuon_Glb_chi2OverDf_MuonCert_Labeled_UL2018C_Reduced",
        "D": "GlbMuon_Glb_chi2OverDf_MuonCert_Labeled_UL2018D_Reduced"
    },
    "eta": {
        "A": "GlbMuon_Glb_eta_MuonCert_Labeled_UL2018A_Reduced",
        "B": "GlbMuon_Glb_eta_MuonCert_Labeled_UL2018B_Reduced",
        "C": "GlbMuon_Glb_eta_MuonCert_Labeled_UL2018C_Reduced",
        "D": "GlbMuon_Glb_eta_MuonCert_Labeled_UL2018D_Reduced"
    },
    "phi": {
        "A": "GlbMuon_Glb_phi_MuonCert_Labeled_UL2018A_Reduced",
        "B": "GlbMuon_Glb_phi_MuonCert_Labeled_UL2018B_Reduced",
        "C": "GlbMuon_Glb_phi_MuonCert_Labeled_UL2018C_Reduced",
        "D": "GlbMuon_Glb_phi_MuonCert_Labeled_UL2018D_Reduced"
    },
    "pt": {
        "A": "GlbMuon_Glb_pt_MuonCert_Labeled_UL2018A_Reduced",
        "B": "GlbMuon_Glb_pt_MuonCert_Labeled_UL2018B_Reduced",
        "C": "GlbMuon_Glb_pt_MuonCert_Labeled_UL2018C_Reduced",
        "D": "GlbMuon_Glb_pt_MuonCert_Labeled_UL2018D_Reduced"
    }
}


# ### Directorio padre

# In[8]:


parent = os.getcwd().split('DQM-DC NMF')[0]+'DQM-DC NMF'