# %% [markdown]
# # Get Snapshots of the Visual Stimulation displayed on the Screen

# %% [markdown]
# ## From a Datafile

# %%
# filename:            /!\ modify if it is not in your "Downloads" folder
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
from physion.analysis.read_NWB import Data

# %%
# load the datafile
data = Data(filename,
            verbose=False)
data.init_visual_stim()

# %% [markdown]
# ### Show a single episode

# %%
episode = 20
data.visual_stim.show_frame(episode) #label={'degree': 20, 'shift_factor': 0.02, 'lw': 2, 'size': 12})

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

# %% [markdown]
# ## From a Protocol

# %%
from physion.visual_stim.build import get_default_params
from physion.visual_stim.stimuli.grating import stim

params = get_default_params('grating')
params['contrast'] = 1
params['radius'] = 200.
params['speed'] = 2.
params['angle'] = 45./2.+90
params['angle-surround'] = 90+27.5
params['contrast-surround'] = 0
params['radius-surround'] = 250.
params['speed-surround'] = 2.
params['units'] = 'lin-deg'
Stim = stim(params)
pt.plt.imshow(np.rot90(Stim.get_image(0), k=3), cmap=pt.binary)
pt.plt.axis('off')
pt.plt.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %%
