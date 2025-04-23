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
# # Compute Orientation Selectivity

# %% [markdown]
# The core functions to analyze the orientation selectivity:
#
# - `selectivity_index`
# - `shift_orientation_according_to_pref`
# - `compute_tuning_response_per_cells`
#
# are implemented in the script [../src/physion/analysis/protocols/orientation_tuning.py](./../src/physion/analysis/protocols/orientation_tuning.py)

# %%
import sys, os
import numpy as np

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src')) # update to your "physion" location
import physion
import physion.utils.plot_tools as pt

from physion.analysis.protocols.orientation_tuning import compute_tuning_response_per_cells

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')

data = physion.analysis.read_NWB.Data(filename, verbose=False)
data.build_dFoF(neuropil_correction_factor=0.9, percentile=10., verbose=False)

Episodes = physion.analysis.process_NWB.EpisodeData(data,
                                                    quantities=['dFoF'],
                                                    protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                                                    verbose=False)

# %% [markdown]
# ## Compute Responses of Visually-Reponsive Cells (i.e. Significantly-Modulated)
#
# N.B. This considers only significantly responsive cells !!

# %%
stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest',                                            
                       positive=True)

response_significance_threshold = 0.001

Tuning = compute_tuning_response_per_cells(data, Episodes,
                                           quantity='dFoF',
                                           stat_test_props=stat_test_props,
                                           response_significance_threshold = response_significance_threshold,
                                           contrast=1,
                                           return_significant_waveforms=True,
                                           verbose=False)

# %% [markdown]
# ## Plot Individual Responses

# %%
fig, AX = pt.figure(axes=(5, int(len(Tuning['Responses'])/5)+1), figsize=(1,.7), hspace=1.5)

for i, ax in enumerate(pt.flatten(AX)):
    if i<len(Tuning['Responses']):
        pt.annotate(ax, 'ROI #%i' % (1+np.arange(data.nROIs)[Tuning['significant_ROIs']][i]), (1,1), va='top', ha='right', fontsize=7)
        pt.annotate(ax, 'SI=%.2f' % Tuning['selectivities'][i], (0,1), fontsize=7)
        ax.plot(Tuning['shifted_angle'], Tuning['Responses'][i], 'k')
        ax.plot(Tuning['shifted_angle'], 0*Tuning['shifted_angle'], 'k:', lw=0.5)
        pt.set_plot(ax, xticks=Tuning['shifted_angle'], 
                    ylabel='(post - pre)\n$\delta$ $\Delta$F/F' if i%5==0 else '',
                    xlabel='angle ($^o$) from preferred orientation' if i==(data.nROIs-1) else '',
                    xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])
    else:
        ax.axis('off')

# %% [markdown]
# ## Single Session Summary

# %%
from scipy import stats
from scipy.optimize import minimize

def func(S, X):
    """ fitting function """
    nS = (S+90)%180-90
    return X[0]*np.exp(-(nS**2/2./X[1]**2))+X[2]


fig, AX = pt.figure(axes=(3, 1), figsize=(1.2, 1), wspace=1.5)

pt.plot(Tuning['shifted_angle'], np.mean(Tuning['Responses'], axis=0), 
        sy=np.std(Tuning['Responses'], axis=0), ax=AX[0])

pt.plot(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses']], axis=0), 
        sy=np.std([r/r[1] for r in Tuning['Responses']], axis=0), ax=AX[1])

pt.scatter(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses']], axis=0), 
        sy=stats.sem([r/r[1] for r in Tuning['Responses']], axis=0), ax=AX[2], ms=3)

# fit
def to_minimize(x0):
    return np.sum((np.mean([r/r[1] for r in Tuning['Responses']], axis=0)-\
                   func(Tuning['shifted_angle'], x0))**2)

res = minimize(to_minimize,
               [0.8, 10, 0.2])
x = np.linspace(-30, 180-30, 100)
AX[2].plot(x, func(x, res.x), lw=2, alpha=.5, color='dimgrey')

pt.set_plot(AX[0], xticks=Tuning['shifted_angle'], 
            ylabel='(post - pre)\n$\delta$ $\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

pt.set_plot(AX[1], xticks=Tuning['shifted_angle'], yticks=np.arange(6)*0.2, ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

pt.set_plot(AX[2], xticks=Tuning['shifted_angle'], yticks=np.arange(3)*0.5, ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])

fig.suptitle(' session: %s ' % os.path.basename(data.filename), fontsize=7)

# %% [markdown]
# ## Multiple Session Summary

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs'))

Responses = []

for f in DATASET['files']:
    print(' - analyzing file: %s  [...] ' % f)
    data = physion.analysis.read_NWB.Data(f, verbose=False)
    data.build_dFoF(neuropil_correction_factor=0.9, percentile=10., verbose=False)

    Episodes = physion.analysis.process_NWB.EpisodeData(data, quantities=['dFoF'],
                                                    protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0], verbose=False)

    Tuning = compute_tuning_response_per_cells(data, Episodes, quantity='dFoF',
                                           stat_test_props=stat_test_props,
                                           response_significance_threshold = response_significance_threshold,
                                           contrast=1)

    Responses.append(np.mean(Tuning['Responses'], axis=0))

# %%
from scipy import stats
from scipy.optimize import minimize

def func(S, X):
    """ fitting function """
    nS = (S+90)%180-90
    return X[0]*np.exp(-(nS**2/2./X[1]**2))+X[2]


fig, ax = pt.figure(figsize=(1.2, 1))


pt.scatter(Tuning['shifted_angle'], np.mean([r/r[1] for r in Responses], axis=0), 
        sy=stats.sem([r/r[1] for r in Responses], axis=0), ax=ax, ms=3)

# fit
def to_minimize(x0):
    return np.sum((np.mean([r/r[1] for r in Responses], axis=0)-\
                   func(Tuning['shifted_angle'], x0))**2)

res = minimize(to_minimize,
               [0.8, 10, 0.2])
x = np.linspace(-30, 180-30, 100)
ax.plot(x, func(x, res.x), lw=2, alpha=.5, color='dimgrey')

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))

pt.set_plot(ax, xticks=Tuning['shifted_angle'], yticks=np.arange(3)*0.5, ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  xlabel='angle ($^o$) from pref.',
            xticks_labels=['%i' % a if (a in [0, 90]) else '' for a in Tuning['shifted_angle'] ])
