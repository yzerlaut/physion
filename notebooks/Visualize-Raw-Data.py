# %% [markdown]
# # Visualize Raw Data

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['../src'] # add src code directory for physion
import physion
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %%
# load a datafile
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')
data = physion.analysis.read_NWB.Data(filename,
                                      verbose=False)
data.build_rawFluo(verbose=False)
data.build_dFoF(verbose=False)

# %% [markdown]
# ## Showing Field of View

# %%
fig, AX = pt.figure(axes=(3,1), 
                    ax_scale=(1.4,3), wspace=0.15)

from physion.dataviz.imaging import show_CaImaging_FOV
#
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[0])
show_CaImaging_FOV(data, key='max_proj', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3, # non-linearity to normalize image
                   ax=AX[1])
show_CaImaging_FOV(data, key='meanImg', 
                   cmap=pt.get_linear_colormap('k', 'tab:green'),
                   NL=3,
                   roiIndex=range(data.nROIs), 
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
fig, AX = \
    plot_raw(data, 
             tlim=[100, data.t_dFoF[-1]], 
             settings=settings)


# %%
