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
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path.append(os.path.join(os.path.expanduser('~'),\
        'work', 'physion', 'src')) # update to your "physion" location

import physion
import physion.utils.plot_tools as pt
pt.set_style('dark-notebook')

from physion.analysis.protocols.orientation_tuning\
          import compute_tuning_response_per_cells, fit_gaussian

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

response_significance_threshold = 0.001 # very very conservative

Tuning = compute_tuning_response_per_cells(data, Episodes,
                                           quantity='dFoF',
                                           stat_test_props=stat_test_props,
                                           response_significance_threshold = response_significance_threshold,
                                           contrast=1,
                                           verbose=False)

# %% [markdown]
# ## Plot Individual Responses

# %%
fig, AX = pt.figure(axes=(5, int(data.nROIs/5)+1), figsize=(1,.7), hspace=1.5)

for i, ax in enumerate(pt.flatten(AX)):
    if i<data.nROIs:
        pt.annotate(ax, ' ROI #%i \n (SI=%.2f)' % (1+i, Tuning['selectivities'][i]), (1,1), 
                    va='center', ha='right', fontsize=7,
                    color='tab:green' if Tuning['significant_ROIs'][i] else 'tab:red')
        # ax.plot(Tuning['shifted_angle'], Tuning['Responses'][i], 'k')
        pt.plot(Tuning['shifted_angle'], Tuning['Responses'][i], 
                sy=Tuning['semResponses'][i], ax=ax)
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
# Gaussian Fit
_, func = fit_gaussian(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses']], axis=0))

# Plot

fig, AX = pt.figure(axes=(3, 1), figsize=(1.2, 1), wspace=1.5)

pt.plot(Tuning['shifted_angle'], np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:], axis=0), 
        sy=np.std(Tuning['Responses'][Tuning['significant_ROIs'],:], axis=0), ax=AX[0])

pt.plot(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        sy=np.std([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), ax=AX[1])

pt.scatter(Tuning['shifted_angle'], np.mean([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        sy=stats.sem([r/r[1] for r in Tuning['Responses'][Tuning['significant_ROIs'],:]], axis=0), 
        ax=AX[2], ms=2)

x = np.linspace(-30, 180-30, 100)

AX[2].plot(x, func(x), lw=2, alpha=.5, color='dimgrey')

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

for contrast in [0.5, 1.0]:

        Tunings = []

        for f in DATASET['files']:
                print(' - analyzing file: %s  [...] ' % f)
                data = physion.analysis.read_NWB.Data(f, verbose=False)
                data.build_dFoF(neuropil_correction_factor=0.9,
                                percentile=10., 
                                verbose=False)

                protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0]
                Episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                                        quantities=['dFoF'], 
                                                                        protocol_name=protocol_name, 
                                                                        verbose=False)

                Tuning = compute_tuning_response_per_cells(data, Episodes, 
                                                        quantity='dFoF', 
                                                        stat_test_props=stat_test_props, 
                                                        response_significance_threshold = response_significance_threshold, 
                                                        contrast=contrast)

                Tunings.append(Tuning)

        # saving data
        np.save(os.path.join(tempfile.tempdir, 
                        'Tunings_contrast-%.1f.npy' % contrast),
                Tunings)

# %%

# loading data
Tunings = np.load(os.path.join(tempfile.tempdir, 'Tunings_contrast-1.0.npy'), 
                  allow_pickle=True)

# mean significant responses per session
Responses = [np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:], 
                     axis=0)
                 for Tuning in Tunings]
# Gaussian Fit
C, func = fit_gaussian(Tunings[0]['shifted_angle'],
                        np.mean([r/r[1] for r in Responses], axis=0))

# Plot
fig, ax = pt.figure(figsize=(1.2, 1))

pt.scatter(Tunings[0]['shifted_angle'], np.mean([r/r[1] for r in Responses], axis=0), 
        sy=stats.sem([r/r[1] for r in Responses], axis=0), ax=ax, ms=3)

x = np.linspace(-30, 180-30, 100)
ax.plot(x, func(x), lw=2, alpha=.5, color='dimgrey')

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))
pt.annotate(ax, 'SI=%.2f' % (1-C[2]), (1., 0.9), ha='right', va='top')

pt.set_plot(ax, xticks=Tunings[0]['shifted_angle'], 
            yticks=np.arange(3)*0.5, 
            ylim=[-0.05, 1.05],
            ylabel='norm. $\delta$ $\Delta$F/F',  
            xlabel='angle ($^o$) from pref.',
            xticks_labels=\
                ['%i' % a if (a in [0, 90]) else ''\
                  for a in Tunings[0]['shifted_angle'] ])

# %%
# --> plot above implemented in the 
from physion.analysis.protocols.orientation_tuning\
        import plot_orientation_tuning_curve

fig, ax = plot_orientation_tuning_curve(\
                                        ['contrast-1.0', 
                                         'contrast-1.0', 
                                         'contrast-0.5'],
                                        path=tempfile.tempdir)
    


# %%
