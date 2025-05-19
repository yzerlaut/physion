# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Visualize Raw Data

# %%
import os

filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

# add the python path:
sys.path.append('../src')
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles

# %%
# load a datafile
#filename = os.path.join(os.path.expanduser('~'), 
#                        'CURATED', 'Pyr-FlexiCortPrelim-FebJuly2022', '2022_07_07-14-45-08.nwb')
data = Data(filename,
            verbose=False)
data.build_rawFluo(verbose=False)
data.build_dFoF(verbose=False)

# %% [markdown]
# ## Showing Field of View

# %%
fig, AX = pt.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)

from physion.dataviz.imaging import show_CaImaging_FOV
#
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=2, # non-linearity to normalize image
                   ax=AX[0])
show_CaImaging_FOV(data, key='max_proj', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=2, # non-linearity to normalize image
                   ax=AX[1])
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=2,
                   roiIndices=range(data.nROIs), 
                   ax=AX[2])

# save on desktop
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'FOV.png'))

# %% [markdown]
# # Show Raw Data

# %%
# default plot
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
settings = find_default_plot_settings(data)
_ = plot_raw(data, settings=settings, tlim=[1200,1300])

# %% [markdown]
# ## Full view

# %%
settings = {'Locomotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': '#1f77b4'},
            'FaceMotion': {'fig_fraction': 1,
                           'subsampling': 1,
                           'color': 'purple'},
            'Pupil': {'fig_fraction': 2,
                      'subsampling': 1,
                      'color': '#d62728'},
             'CaImaging': {'fig_fraction': 10,
                           'subsampling': 1,
                           'subquantity': 'dF/F',
                           'roiIndices': np.random.choice(np.arange(data.nROIs), np.min([20,data.nROIs]), replace=False),
                           'color': '#2ca02c'}
           }
plot_raw(data, tlim=[100, data.t_dFoF[-1]], settings=settings)
