# %% [markdown]
# # Compute Contrast Sensitivity

# %% [markdown]
# The core functions to analyze the orientation selectivity:
#
# - `selectivity_index`
# - `shift_orientation_according_to_pref`
# - `compute_tuning_response_per_cells`
#
# are implemented in the script [src/physion/analysis/protocols/contrast_sensitivity.py](../../src/physion/analysis/protocols/contrast_sensitivity.py)

# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path.append('../../src') # add src code directory for physion

import physion.utils.plot_tools as pt
pt.set_style('dark')

import physion
from physion.analysis.protocols.contrast_sensitivity\
          import compute_sensitivity_per_cells

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2024_08_27-11-16-53.nwb')

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

Sensitivity = compute_sensitivity_per_cells(data, Episodes,
                                        quantity='dFoF',
                                        stat_test_props=stat_test_props,
                                        response_significance_threshold = response_significance_threshold,
                                        angle=0, # 0 or 90 here !
                                        verbose=False)

# %% [markdown]
# ## Plot Individual Responses

# %%
fig, AX = pt.figure(axes=(5, int(data.nROIs/5)+1), ax_scale=(1,.7), hspace=1.5)

for i, ax in enumerate(pt.flatten(AX)):
    if i<data.nROIs:
        pt.annotate(ax, ' ROI #%i ' % (1+i), (1,1), 
                    va='center', ha='right', fontsize=7,
                    color='tab:green' if Sensitivity['significant_ROIs'][i] else 'tab:red')
        pt.plot(Sensitivity['contrast'], Sensitivity['Responses'][i], 
                sy=Sensitivity['semResponses'][i], ax=ax)
        ax.plot(Sensitivity['contrast'], 0*Sensitivity['contrast'], 'k:', lw=0.5)
        pt.set_plot(ax, 
                    xticks=Sensitivity['contrast'], 
                    ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F' if i%5==0 else '',
                    xlabel='contrast' if i==(data.nROIs-1) else '',
                    xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])
    else:
        ax.axis('off')

# %% [markdown]
# ## Single Session Summary

# %%

# Plot

fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1), wspace=1.5)

pt.plot(Sensitivity['contrast'], np.mean(Sensitivity['Responses'][Sensitivity['significant_ROIs'],:], axis=0), 
        sy=np.std(Sensitivity['Responses'][Sensitivity['significant_ROIs'],:], axis=0), ax=AX[0])

pt.scatter(Sensitivity['contrast'], np.mean([r for r in Sensitivity['Responses'][Sensitivity['significant_ROIs'],:]], axis=0), 
        sy=stats.sem([r for r in Sensitivity['Responses'][Sensitivity['significant_ROIs'],:]], axis=0), 
        ax=AX[1], ms=2)

x = np.linspace(-30, 180-30, 100)

pt.set_plot(AX[0], xticks=Sensitivity['contrast'], 
            ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F',  
            xlabel='contrast',
            xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])

pt.set_plot(AX[1], 
            ylabel='$\\delta$ $\\Delta$F/F',
            xlabel='contrast',
            xticks=Sensitivity['contrast'], 
            xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])

fig.suptitle(' session: %s ' % os.path.basename(data.filename), fontsize=7);

# %% [markdown]
# ## Multiple Session Summary

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs'),
        for_protocols=['ff-gratings-2orientations-8contrasts-15repeats',
                       'ff-gratings-2orientations-8contrasts-15repeats'])
print(DATASET)

for angle in [0, 90]:

        Sensitivities = []

        for f in DATASET['files']:
                print(' - analyzing file: %s  [...] ' % f)
                data = physion.analysis.read_NWB.Data(f, verbose=False)
                nROIs_original = data.nROIs # manually-selected ROIs
                data.build_dFoF(neuropil_correction_factor=0.9,
                                percentile=10., 
                                verbose=False)
                nROIs_final = data.nROIs # ROIs after dFoF criterion

                protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0]
                Episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                                        quantities=['dFoF'], 
                                                                        protocol_name=protocol_name, 
                                                                        verbose=False)

                Sensitivity = compute_sensitivity_per_cells(data, Episodes, 
                                                            quantity='dFoF', 
                                                            stat_test_props=stat_test_props, 
                                                            response_significance_threshold = response_significance_threshold, 
                                                            angle=angle)
                Sensitivity['nROIs_original'] = nROIs_original
                Sensitivity['nROIs_final'] = nROIs_final
                Sensitivity['nROIs_responsive'] = np.sum(Sensitivity['significant_ROIs'])

                Sensitivities.append(Sensitivity)

        # saving data
        np.save(os.path.join(tempfile.tempdir, 
                        'Sensitivities_WT_angle-%.1f.npy' % angle),
                Sensitivities)

# %%

# loading data
Sensitivities = np.load(os.path.join(tempfile.tempdir, 'Sensitivities_WT_angle-0.0.npy'), 
                  allow_pickle=True)

# mean significant responses per session
Responses = [np.mean(Tuning['Responses'][Tuning['significant_ROIs'],:], 
                     axis=0)
                 for Tuning in Sensitivities]

# Plot
fig, ax = pt.figure(ax_scale=(1.2, 1))

pt.scatter(Sensitivities[0]['contrast'], np.mean(Responses, axis=0), 
        sy=stats.sem(Responses, axis=0), ax=ax, ms=3)

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))

pt.set_plot(ax, 
            ylabel='$\delta$ $\Delta$F/F',  
            xlabel='contrast',
            xticks=np.arange(3)*0.5)

# %%
# --> plot above implemented in the orientation_tuning protocol
import tempfile
from physion.analysis.protocols.contrast_sensitivity\
        import plot_contrast_sensitivity

fig, ax = plot_contrast_sensitivity(\
                        ['WT_contrast-1.0', 
                         'WT_contrast-1.0', 
                         'WT_contrast-0.5'],
                        #  average_by='ROIs',
                        #  using='fit',
                        path=tempfile.tempdir)
    

fig, ax = plot_orientation_tuning_curve(\
                                        ['WT_contrast-1.0', 
                                         'WT_contrast-1.0', 
                                         'WT_contrast-0.5'],
                                        #  average_by='ROIs',
                                        path=tempfile.tempdir)
    


# %%

# %%
