# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Convert NWB files to pandas Dataframes

# %%
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src')) # update to your "physion" location
import physion
import physion.utils.plot_tools as pt

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'NDNF-WT', 'NWBs',
                        '2022_12_14-13-27-41.nwb')
data = physion.analysis.dataframe.NWB_to_dataframe(filename,
                                                   visual_stim_features='per-protocol',
                                                   #visual_stim_label='per-protocol-and-parameters',
                                                   #visual_stim_label='per-protocol-and-parameters-and-timepoints', #
                                                   subsampling = 10,
                                                   verbose=False)

# %%
def min_max(array):
    return (array-array.min())/(array.max()-array.min())

def color(key):
    if 'Pupil' in key:
        return pt.plt.cm.Set1(0)
    elif 'Gaze' in key:
        return pt.plt.cm.Set1(4)
    elif 'Running' in key:
        return pt.plt.cm.Set1(1)
    elif 'Whisking' in key:
        return pt.plt.cm.Set1(3)
    elif 'VisStim' in key:
        return pt.plt.cm.Greys(np.random.uniform(0.2, .6))
    else:
        return pt.plt.cm.Greens(np.random.uniform(0.5, .8))
    
fig, ax = pt.plt.subplots(figsize=(8,10))
i = 0
for key in data.keys():
    if key !='time':
        c = color(key)
        ax.plot(data['time'], -i+.8*min_max(data[key].astype(float)), color=c, lw=1) # convert bool to float when needed
        ax.annotate(key.replace('_', '  \n').replace('-', '  \n')+' ', (0.5, -i+.1), ha='right', va='center', color=c, fontsize=5)
        i+=1
                
ax.axis('off');

# %%
