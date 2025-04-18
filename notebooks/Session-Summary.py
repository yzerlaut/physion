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
# # Session Summary 

# %%
filename = '/Users/yann/ASSEMBLE/2024_03_08-10-59-53.nwb'

# %% [markdown]
# ## Load Datafile

# %%
import physion, os
data = physion.analysis.read_NWB.Data(filename,
                                      verbose=False)

# %% [markdown]
# ## Show Field of View

# %%
figFOV, AX = physion.utils.plot_tools.figure(axes=(3,1), figsize=(1.4,3), wspace=0.15)

from physion.dataviz.imaging import show_CaImaging_FOV
#
show_CaImaging_FOV(data, key='meanImg', 
                   NL=2, # non-linearity to normalize image
                   ax=AX[0])
show_CaImaging_FOV(data, key='max_proj', 
                   NL=2, # non-linearity to normalize image
                   ax=AX[1])
show_CaImaging_FOV(data, key='meanImg', 
                   NL=2,
                   roiIndices=range(data.nROIs), 
                   ax=AX[2])
# save on desktop
#figFOV.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'FOV.png'))

# %% [markdown]
# ## Raw Data -- Full View 

# %%
settings = physion.dataviz.raw.find_default_plot_settings(data, with_subsampling=True)
# settings['CaImaging']['roiIndices'] = [1, 13, 0, 34, 5, 6, 8]
figRaw, _ = physion.dataviz.raw.plot(data, tlim=[0,data.tlim[1]], settings=settings)

# %% [markdown]
# ## Raw Data -- Zoomed View 

# %%
zoom = [100,160] 
settings = physion.dataviz.raw.find_default_plot_settings(data)
# settings['CaImaging']['roiIndices'] = [1, 13, 0, 34, 5, 6, 8]
figRaw, _ = physion.dataviz.raw.plot(data, tlim=zoom, settings=settings)

# %%
zoom = [620,680] 
settings = physion.dataviz.raw.find_default_plot_settings(data)
# settings['CaImaging']['roiIndices'] = [1, 13, 0, 34, 5, 6, 8]
figRaw, _ = physion.dataviz.raw.plot(data, tlim=zoom, settings=settings)

# %% [markdown]
# ## Raw Data -- Full View -- All ROIs

# %%
from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
settings = physion.dataviz.raw.find_default_plot_settings(data, with_subsampling=True)
settings['CaImaging']['roiIndices'] = range(data.nROIs)
settings['CaImaging']['fig_fraction']=10.
figRaw, _ = plot_raw(data, figsize=(9,15),
                     tlim=[0,data.tlim[1]], settings=settings)

# %%
## 

# %%
