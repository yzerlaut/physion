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
# # Get Snapshots of the Visual Stimulation displayed on the Screen

# %%
# filename:            /!\ modify if it is not in your "Downloads" folder
import os
filename = os.path.join(os.path.expanduser('~'),'CURATED','SST-FF-Gratings-Stim','Wild-Type', '2023_05_11-15-25-41.nwb')

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

# add the python path:
sys.path.append('../src')
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data

import warnings
warnings.filterwarnings("ignore") # disable the UserWarning from pynwb (arrays are not well oriented)

# %%
# load the datafile
data = Data(filename,
            verbose=False)
data.init_visual_stim()

# %% [markdown]
# ## Show a single episode

# %%
episode = 1
data.visual_stim.show_frame(episode, label={'degree': 20, 'shift_factor': 0.02, 'lw': 2, 'fontsize': 12})

# %%
# or:
episode = 0
data.visual_stim.plot_stim_picture(episode)

# %% [markdown]
# ## Show several episodes

# %%
episodes = np.arange(16)

fig, AX = pt.figure(axes=(len(episodes), 1), figsize=(2,2))
for e, episode in enumerate(episodes):
    data.visual_stim.plot_stim_picture(episode, ax=AX[e])
fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'stim.pdf'))

# %%
