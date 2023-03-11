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
import sys, pathlib, os, itertools
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

# add the python path:
sys.path.append('../../src')
from physion.analysis.read_NWB import Data
from physion.analysis.dataframe import NWB_to_dataframe, extract_stim_keys
from physion.utils import plot_tools as pt

subsampling = 2 


# %% [markdown]
# ## Split training and test sets

# %%
filename = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022', '2022_12_14-13-27-41.nwb')
data = Data(filename)
df = NWB_to_dataframe(filename,
                      visual_stim_label='per-protocol-and-parameters',
                      subsampling = subsampling ,
                      verbose=False)

# %%
from sklearn.model_selection import StratifiedKFold

# %%
indices = np.arange(len(df['time']))

fig, ax = plt.subplots(figsize=(7,2))

###################################
# first spontaneous activity
spont_cond = df['VisStim-grey-10min']
Nspont = int(np.sum(spont_cond)/2)

spont_train_sets = [indices[spont_cond][:Nspont],
                    indices[spont_cond][Nspont:]]
spont_test_sets = [indices[spont_cond][Nspont:],
                   indices[spont_cond][:Nspont]]

###################################
# then stimulus evoked activity
stim_keys = [k for k in df if ('VisStim' in k)]
stimID = 0*df['time']
for i, k in enumerate(stim_keys):
    stimID[df[k]] = i+1
stim_cond = (~df['VisStim-grey-10min']) & (stimID>0)

stim_train_sets, stim_test_sets = StratifiedKFold(2).split(X=np.random.randn(len(stimID)), 
                                                           y=stimID, groups=stimID)
stim_train_sets = [np.intersect1d(indices[stim_cond], s) for s in stim_train_sets]
stim_test_sets = [np.intersect1d(indices[stim_cond], s) for s in stim_test_sets]


###################################
# now plotting

ii = 10
for train, test in zip(spont_train_sets, spont_test_sets):
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=10)
    ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=10)
    ii-=2
ax.annotate('spont. act. ', (0,ii+3), ha='right')
ii-=1

for train, test in zip(stim_train_sets, stim_test_sets):
    ax.scatter(df['time'][train], ii+0.5+np.zeros(len(df['time'][train])), c='tab:red', marker="_", lw=10)
    ax.scatter(df['time'][test], ii+0.5+np.zeros(len(df['time'][test])), c='tab:blue', marker="_", lw=10)
    ii-=2
ax.annotate('stim. evoked ', (0,ii+3), ha='right')

ax.scatter(
    df['time'][stim_cond], [ii-0.5] * np.sum(stim_cond), c=stimID[stim_cond], marker="_", lw=10, 
    cmap=plt.cm.tab20)
ax.annotate('stim ID  ', (0,ii-1), ha='right')


ax.annotate('training set', (1,1), color='tab:red', xycoords='axes fraction')
ax.annotate('test set\n', (1,1), color='tab:blue', xycoords='axes fraction')

ax.axis('off')
ax.set_xlabel("time (s)")
ax.set_title('cross-validation (linewidth makes proportions misleading)')
ax.axes.get_xaxis().set_visible(True)

# %% [markdown]
# ## Running the model on the training set

# %%
# Re-loeading the data with the proper annotation !

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning) # 

filename = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022', '2022_12_14-13-27-41.nwb')
data = Data(filename)
df = NWB_to_dataframe(filename,
                      visual_stim_label='per-protocol-and-parameters-and-timepoints',
                      subsampling = subsampling,
                      verbose=False)
stim_keys = [k for k in df if 'VisStim' in k]

# %%
# simple linear regression
from sklearn.linear_model import LinearRegression, Ridge

iTrainTest = 0
roiIndex = 2
train_indices = stim_train_sets[iTrainTest]
#reg = LinearRegression().fit(df[stim_keys].loc[train_indices,:], 
#                             df['dFoF-ROI%i'%roiIndex].loc[train_indices])
reg = Ridge().fit(df[stim_keys].loc[train_indices,:], 
                             df['dFoF-ROI%i'%roiIndex].loc[train_indices])

# %%
test_indices = stim_test_sets[iTrainTest]
real, predicted = 0*df['time'], 0*df['time'] # need to have the full shape for the plot later

predicted[test_indices] = reg.predict(df[stim_keys].loc[test_indices,:])
real[test_indices] = df['dFoF-ROI%i'%roiIndex].loc[test_indices]

# %%
fig, ax = plt.subplots(figsize=(7,1))
#ax.axis('off')
ax.plot(df['time'], predicted, label='predicted', color='tab:red')
ax.plot(df['time'], real, label='real')
ax.legend()
#ax.set_ylim([0,5])
ax.set_xlim([0, 200])

# %%
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol
fig, AX = pt.plt.subplots(5, len(protocols),
                          figsize=(7,5))
pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

STIM = extract_stim_keys(df, indices_subset=stim_test_sets[iTrainTest])

for p, protocol in enumerate(protocols):
    
    varied_keys = [k for k in STIM[protocol] if k not in ['times', 'DF-key']]
    varied_values = [np.sort(np.unique(STIM[protocol][k])) for k in varied_keys]

    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')
    
    i=0
    for values in itertools.product(*varied_values):
       
        stim_cond = np.ones(len(STIM[protocol]['times']), dtype=bool)
        for k, v in zip(varied_keys, values):
            stim_cond = stim_cond & (STIM[protocol][k]==v)
        
        iTime_sorted = np.argsort(STIM[protocol]['times'][stim_cond])
        times = STIM[protocol]['times'][stim_cond][iTime_sorted]
        resp, pred = [], []

        for t in iTime_sorted:
            stim_time_cond = df[STIM[protocol]['DF-key'][stim_cond][t]]
            resp.append(np.mean(real[stim_time_cond]))
            pred.append(np.mean(predicted[stim_time_cond]))

        AX[i][p].plot(times, resp, color='tab:blue')
        AX[i][p].plot(times, pred, color='tab:red')

        AX[i][p].annotate('%s=%s' % (varied_keys, values),
                          (0,-0.1), fontsize=4,
                          rotation=90, ha='right',
                          #va='top', ha='center',
                          xycoords='axes fraction')
        
        i+=1
        
AX[-1][0].annotate('predicted', (0,0), color='tab:red', xycoords='axes fraction', fontsize=9)
AX[-1][0].annotate('real\n', (0,0), color='tab:blue', xycoords='axes fraction', fontsize=9)
        
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    if np.isfinite(ax.dataLim.x0):
        pt.draw_bar_scales(ax,
                           Xbar=1., Xbar_label='1s',
                           Ybar=1., Ybar_label='1$\Delta$F/F', fontsize=7)
pt.set_common_xlims(AX)

# %% [markdown]
# ## Control
#
# ### checking that the data re-arrange in dataframes per stimulus and timestamps (after interpolation) give the same results than trial-averaging based on intervals

# %%
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol
fig, AX = pt.plt.subplots(5, len(protocols),
                          figsize=(7,5))
pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

roiIndex = 2
dFoF = df['dFoF-ROI%i' % roiIndex]

STIM = extract_stim_keys(df)

for p, protocol in enumerate(protocols):
    
    varied_keys = [k for k in STIM[protocol] if k not in ['times', 'DF-key']]
    varied_values = [np.sort(np.unique(STIM[protocol][k])) for k in varied_keys]

    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')
    
    i=0
    for values in itertools.product(*varied_values):
       
        stim_cond = np.ones(len(STIM[protocol]['times']), dtype=bool)
        for k, v in zip(varied_keys, values):
            stim_cond = stim_cond & (STIM[protocol][k]==v)
        
        iTime_sorted = np.argsort(STIM[protocol]['times'][stim_cond])
        times = STIM[protocol]['times'][stim_cond][iTime_sorted]
        mean, std = [], []

        for t in iTime_sorted:
            stim_time_cond = df[STIM[protocol]['DF-key'][stim_cond][t]]
            mean.append(np.mean(dFoF[stim_time_cond]))
            std.append(np.std(dFoF[stim_time_cond]))

        #AX[i][p].plot(times, resp, 'k-')
        pt.plot(times, mean, std, ax=AX[i][p])

        AX[i][p].annotate('%s=%s' % (varied_keys, values),
                          (0,-0.1), fontsize=4,
                          rotation=90, ha='right',
                          #va='top', ha='center',
                          xycoords='axes fraction')
        
        i+=1
        
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    if np.isfinite(ax.dataLim.x0):
        pt.draw_bar_scales(ax,
                           Xbar=1., Xbar_label='1s',
                           Ybar=1., Ybar_label='1$\Delta$F/F', fontsize=7)
pt.set_common_xlims(AX)

# %%
