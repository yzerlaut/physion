# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

# add the python path:
sys.path.append('../../src')
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.dataviz.raw import plot as plot_raw

# %%
# load a datafile
filename = os.path.join(os.path.expanduser('~'), 
                        'CURATED', 'Pyr-FlexiCortPrelim-FebJuly2022', '2022_07_07-14-45-08.nwb')
data = Data(filename,
            verbose=False)
data.build_dFoF(verbose=False)

# %% [markdown]
# ## Showing Field of View

# %%
from physion.dataviz.imaging import show_CaImaging_FOV
show_CaImaging_FOV(data, key='meanImg', NL=10, roiIndices=range(data.nROIs))

# %% [markdown]
# ## Showing visually evoked activity

# %%
tlim = [38,88]

fig, _ = plot_raw(data, tlim, 
                  settings={'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='tab:green',
                                             roiIndices=[6]+\
                                                 list(np.random.choice(range(data.vNrois), 7))),
                            'VisualStim':dict(fig_fraction=0, color='black')},
                            Tbar=1)
#fig.savefig(os.path.join(os.path.expanduser('~'), 'Desktop', 'fig.svg'))

# %% [markdown]
# ## Showing visually-evoked + behavior

# %%
tlim = [40,300]

fig, _ = plot_raw(data, tlim, 
                  settings={'Locomotion':dict(fig_fraction=1, subsampling=1, color='tab:blue'),
                            'FaceMotion':dict(fig_fraction=1, subsampling=1, color='tab:purple'),
                            'Pupil':dict(fig_fraction=1, subsampling=1, color='tab:red'),
                            'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             subquantity='dF/F', color='tab:green',
                                             #roiwIndices=[0,1,2,3,4,5,6,7]),
                                             roiIndices=np.random.choice(range(data.vNrois),4)),
                            'CaImagingRaster':dict(fig_fraction=2, subsampling=1,
                                                   roiIndices='all',
                                                   normalization='per-line',
                                                   subquantity='dF/F'),
                            'VisualStim':dict(fig_fraction=0, color='black')},
                            Tbar=5, figsize=(2,4))
#ge.save_on_desktop(fig)


# %%
