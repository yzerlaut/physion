
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
# # Ocular Dominance maps
# 
# from:
# 
# > Cang, J., Kalatsky, V. A., Löwel, S., & Stryker, M. P. (2005). 
# >
# > Optical imaging of the intrinsic signal as a measure of cortical plasticity in the mouse. 
# >
# > Visual neuroscience, 22(5), 685-691.

# %%

import sys
import numpy as np
sys.path.append('../src')
from physion.intrinsic.ocular_dominance import make_fig

maps = np.load(os.path.join(os.path.expanduser('~'), 'DATA',
                            '2025_06_26', '07-58-35', 
                            'ocular-dominance-maps.npy'),
                            allow_pickle=True).item()
fig, AX = make_fig(maps)
# %%
