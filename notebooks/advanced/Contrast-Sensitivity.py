# %% [markdown]
# # Compute Contrast Sensitivity

# %% [markdown]
# The core functions to analyze the orientation selectivity:
#
# - `compute_sensitivity_per_cells`
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
data.build_dFoF(neuropil_correction_factor=0.9, 
                percentile=10., 
                verbose=False)

# %%
from physion.dataviz.raw import plot

settings = {\
 'Locomotion': {'fig_fraction': 1, 'subsampling': 1, 'color': '#1f77b4'},
 'FaceMotion': {'fig_fraction': 1, 'subsampling': 1, 'color': 'purple'},
#  'GazeMovement': {'fig_fraction': 0.5, 'subsampling': 1, 'color': '#ff7f0e'},
 'Pupil': {'fig_fraction': 1, 'subsampling': 1, 'color': '#d62728'},
 'CaImaging': {'fig_fraction': 4, 'subsampling': 1, 'subquantity': 'dFoF',
  'color': '#2ca02c',
  'roiIndices': np.random.choice(np.arange(data.nROIs), 10)},
 'CaImagingRaster': {'fig_fraction': 2, 'subsampling': 1,
  'roiIndices': 'all', 'normalization': 'per-line', 'subquantity': 'dF/F'},
 'VisualStim': {'fig_fraction': 0.2, 'color': 'black'}}
plot(data, tlim=[100,250], 
     settings=settings,
     figsize=(12,8))

# %%
from physion.dataviz.imaging import show_CaImaging_FOV
show_CaImaging_FOV(data, NL=4)

# %%
Episodes = physion.analysis.process_NWB.EpisodeData(data,
                                                    quantities=['dFoF'],
                                                    protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                                                    verbose=False)

# %% [markdown]
# ## Compute Responses of Cells (all, not only signtificantly-modulated)

# %%
stat_test_props = dict(interval_pre=[-1.,0],                                   
                       interval_post=[1.,2.],                                   
                       test='ttest') # no need to set "sign", compute_sensitivity tests both !                                          

response_significance_threshold = 0.01 # very very conservative

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
                    color='tab:green' if np.sum(Sensitivity['significant_pos'][i,:]) else 'tab:red')
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

# Plot Response Levels
fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1), wspace=1.5, top=2.)

pt.plot(Sensitivity['contrast'], np.mean(Sensitivity['Responses'], axis=0), 
        sy=np.std(Sensitivity['Responses'], axis=0), ax=AX[0])

pt.plot(Sensitivity['contrast'], 
        np.mean(Sensitivity['Responses'][\
                        np.sum(Sensitivity['significant_pos'],axis=1)>0, :],
                axis=0), 
        sy=np.std(Sensitivity['Responses'][\
                        np.sum(Sensitivity['significant_pos'],axis=1)>0, :],
                axis=0), 
        ax=AX[1])
pt.set_common_ylims(AX)

pt.set_plot(AX[0], title='All (n=%i ROIs)' % data.nROIs,
            xticks=Sensitivity['contrast'], 
            ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F',  
            xlabel='contrast',
            xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])

pt.set_plot(AX[1], 
            title='+Responsive-Only (n=%i ROIs)'\
                 % np.sum(np.sum(Sensitivity['significant_pos'],axis=1)>0),
            ylabel='$\\delta$ $\\Delta$F/F',
            xlabel='contrast',
            xticks=Sensitivity['contrast'], 
            xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])

fig.suptitle(' session: %s ' % os.path.basename(data.filename), fontsize=7);

# %%
# Plot Responsiveness

fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1), wspace=1.5, top=2.)

##########################
### Positive Responses ### 
##########################
for i, c in enumerate(Sensitivity['contrast']):
        AX[0].bar([c], 
                  [100*np.sum(Sensitivity['significant_pos'][:,i],axis=0)/data.nROIs],
                  color='tab:green', width=0.08)

##########################
### Negative Responses ### 
##########################
for i, c in enumerate(Sensitivity['contrast']):
        AX[1].bar([c], 
                  [100*np.sum(Sensitivity['significant_neg'][:,i],axis=0)/data.nROIs],
                  color='tab:red', width=0.08)

pt.set_common_ylims(AX)
for ax, label in zip(AX, ['positive', 'negative']):
        pt.set_plot(ax, title='%s-resp.' % label,
                ylabel='% responsive',
                xlabel='contrast',
                xticks=Sensitivity['contrast'], 
                xticks_labels=['%.2f' % c if j%3==1 else '' for j, c in enumerate(Sensitivity['contrast'])])
fig.suptitle(' session: %s (n=%i ROIs)' % (
                                os.path.basename(data.filename), data.nROIs),
                fontsize=7)

# %% [markdown]
# ## Multiple Session Summary

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs'),
        for_protocols=['ff-gratings-2orientations-8contrasts-15repeats',
                       'ff-gratings-2orientations-8contrasts-15repeats'])

dFoF_options = dict(\
    method_for_F0='percentile',
    percentile=5.,
    roi_to_neuropil_fluo_inclusion_factor=1.15,
    neuropil_correction_factor=0.8)

for angle in [0, 90]:

        Sensitivities = []

        for f in DATASET['files']:
                print(' - analyzing file: %s  [...] ' % f)
                data = physion.analysis.read_NWB.Data(f, verbose=False)
                nROIs_original = data.nROIs # manually-selected ROIs
                data.build_dFoF(**dFoF_options)
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
                Sensitivity['subject'] = data.nwbfile.subject.subject_id

                Sensitivities.append(Sensitivity)

        # saving data
        np.save(os.path.join(tempfile.tempdir, 
                        'Sensitivities_WT_angle-%.1f.npy' % angle),
                Sensitivities)

# %%

# loading data
Sensitivities = np.load(os.path.join(tempfile.tempdir, 
                                     'Sensitivities_WT_angle-0.0.npy'), 
                  allow_pickle=True)

# mean significant responses per session
Responses = [np.mean(S['Responses'], axis=0) for S in Sensitivities]

# Plot
fig, ax = pt.figure(ax_scale=(1.2, 1))

pt.plot(Sensitivities[0]['contrast'], np.mean(Responses, axis=0), 
        sy=stats.sem(Responses, axis=0), ax=ax, ms=3)

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))

pt.set_plot(ax, 
            ylabel='$\delta$ $\Delta$F/F',  
            xlabel='contrast',
            xticks=np.arange(3)*0.5)

# %%
# --> plot above implemented in the orientation_tuning protocol
import tempfile, sys

sys.path += ['../../src'] # add src code directory for physion
from physion.analysis.protocols.contrast_sensitivity\
        import plot_contrast_sensitivity, plot_contrast_responsiveness

fig, ax = plot_contrast_sensitivity(\
                        ['WT_angle-0.0', 
                         'WT_angle-90.0'],
                        #  average_by='ROIs',
                         average_by='subjects',
                        path=tempfile.tempdir)

fig, ax = plot_contrast_responsiveness(\
                        ['WT_angle-0.0', 
                         'WT_angle-90.0'],
                         sign='negative',
                         nROIs='final', # "original" or "final", before/after dFoF criterion
                        path=tempfile.tempdir)

fig, ax = plot_contrast_responsiveness(\
                        ['WT_angle-0.0', 
                         'WT_angle-90.0'],
                         sign='positive',
                         nROIs='final', # "original" or "final", before/after dFoF criterion
                        path=tempfile.tempdir)
    
# %%
