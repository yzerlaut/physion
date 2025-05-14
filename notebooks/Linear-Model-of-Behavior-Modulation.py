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
# # Linear Model for the Behavioral Modulation of Neural Activity

# %%
import sys, os
import numpy as np
from sklearn.linear_model import Ridge

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src')) # update to your "physion" location
import physion
import physion.utils.plot_tools as pt

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')

data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   visual_stim_features='',
                                                   subsampling = 2,
                                                   verbose=False)

data2 = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   visual_stim_features='',
                                                   add_shifted_behavior_features=True,
                                                   subsampling = 2,
                                                   verbose=False)

# %%
bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]

N = 10
fig, AX = pt.figure((1,N), figsize=(3,1), hspace=0.2)
pt.annotate(AX[0], 'Ridge model\n', (1,1), ha='right', color='b')
pt.annotate(AX[0], 'with delayed features', (1,1), ha='right', color='r')
for i in range(N):
    
    AX[i].plot(data['time'], data['dFoF-ROI%i' % i], 'g-')
    
    bhv_keys = [k for k in data.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]
    lin = Ridge(alpha=0.1).fit(data[bhv_keys], data['dFoF-ROI%i' % i])
    AX[i].plot(data['time'], lin.predict(data[bhv_keys]), 'b-', lw=0.5)
    pt.annotate(AX[i], '%.1f%%' % (100*lin.score(data[bhv_keys], data['dFoF-ROI%i' % i])), (0,1), color='b', va='top', fontsize=6)
    
    bhv_keys = [k for k in data2.keys() if (('Run' in k) or ('Gaze' in k) or ('Whisk' in k) or ('Pupil' in k))]
    lin2 = Ridge(alpha=0.1).fit(data2[bhv_keys], data2['dFoF-ROI%i' % i])
    AX[i].plot(data2['time'], lin2.predict(data2[bhv_keys]), 'r-', lw=0.2)
    pt.annotate(AX[i], '\n%.1f%%' % (100*lin2.score(data2[bhv_keys], data2['dFoF-ROI%i' % i])), (0,1), color='r', va='top', fontsize=6)
    
    
    pt.set_plot(AX[i], ['left', 'bottom'] if i==(N-1) else ['left'], ylabel='$\\Delta$F/F\n'+'ROI-%i' % i)

# %%
