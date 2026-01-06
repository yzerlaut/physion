# %% [markdown]
# # Analyze Temporal Dynamics

# %%
import sys, os, tempfile
import numpy as np
from scipy import stats

sys.path.append('../../src') # add src code directory for physion

import physion.utils.plot_tools as pt
pt.set_style('dark')

import physion

dFoF_parameters = dict(\
        roi_to_neuropil_fluo_inclusion_factor=1.15,
        neuropil_correction_factor = 0.7,
        # method_for_F0 = 'sliding_percentile',
        method_for_F0 = 'percentile',
        percentile=5., # percent
        sliding_window = 5*60, # seconds
)
TAU_DECONVOLUTION = 0.8

stat_test_props=dict(interval_pre=[-1.,0],                                   
                   interval_post=[0.,1.],                                   
                   test='anova',                                            
                   sign='positive')

response_significance_threshold=5e-2

contrast = 1.0
orientation = 0.

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2024_08_27-11-16-53.nwb')

data = physion.analysis.read_NWB.Data(filename, verbose=False)
data.build_dFoF(**dFoF_parameters)

# %% [markdown]
# ## Compute Deconvolved Responses of all Cells 

# %%
data.build_Deconvolved(Tau=TAU_DECONVOLUTION)

Episodes = physion.analysis.episodes.build.EpisodeData(data,
                                                       quantities=['Deconvolved'],
                                                       protocol_name=[p for p\
                                                         in data.protocols if 'ff-gratings' in p][0],
                                                       verbose=False)
epCond = Episodes.find_episode_cond(key=['contrast', 'angle'], 
                                    value=[contrast, orientation])

#
# %% [markdown]
# ## Plot Individual Responses

# %%
fig, AX = pt.figure(axes=(5, int(data.nROIs/5)+1), ax_scale=(1,.7), hspace=1.5)

significant = np.zeros(data.nROIs, dtype=bool)
for i, ax in enumerate(pt.flatten(AX)):
    if i<data.nROIs:
        cell_resp = Episodes.compute_summary_data(stat_test_props,
                                                  response_args=dict(roiIndex=i))
        cond = (cell_resp['angle']==orientation) &\
                        (cell_resp['contrast']==contrast)
        significant[i] = cell_resp['significant'][cond][0]
        pt.annotate(ax, ' ROI #%i ' % (1+i), (1,1), 
                    color='tab:green' if cell_resp['significant'][cond] else 'tab:red',
                    va='center', ha='right', fontsize=7)
        pt.plot(Episodes.t, 
                np.mean(Episodes.Deconvolved[epCond,i,:], axis=0),
                sy = 0*np.std(Episodes.Deconvolved[epCond,i,:], axis=0), ax=ax)
        pt.set_plot(ax, 
                    ylabel='(post - pre)\n$\\delta$ $\\Delta$F/F' if i%5==0 else '',
                    xlabel='time (s)' if i==(data.nROIs-1) else '')
    else:
        ax.axis('off')

# %% [markdown]
# ## Single Session Summary

# %%

# Plot Response Levels
fig, AX = pt.figure(axes=(2, 1), ax_scale=(1.2, 1), wspace=1.5, top=2.)

pt.plot(Episodes.t, 
        np.mean(Episodes.Deconvolved[epCond,:,:], axis=(0,1)),
        sy = stats.sem(np.mean(Episodes.Deconvolved[epCond,:,:], axis=0), axis=0), 
        ax=AX[0])

pt.set_plot(AX[0], title='All (n=%i ROIs)' % data.nROIs,
            ylabel='(post - pre)\n Deconv. $\\Delta$F/F',  
            xlabel='time (s)')

pt.plot(Episodes.t, 
        Episodes.Deconvolved[epCond,:,:].mean(axis=0)[significant,:].mean(axis=0),
        sy = stats.sem(Episodes.Deconvolved[epCond,:,:].mean(axis=0), axis=0), 
        ax=AX[1])

pt.set_plot(AX[1], title='+Responsive (n=%i ROIs)' % np.sum(significant),
            ylabel='(post - pre)\n Deconv. $\\Delta$F/F',  
            xlabel='time (s)')

pt.set_common_ylims(AX)

fig.suptitle(' session: %s ' % os.path.basename(data.filename), fontsize=7);

# %% [markdown]
# ## Multiple Session Summary

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs'),
        for_protocols=['ff-gratings-8orientation-2contrasts-10repeats',
                       'ff-gratings-8orientations-2contrasts-15repeats'])
for angle in [0, 90]:

        Responses = []

        for f in DATASET['files']:
                print(' - analyzing file: %s  [...] ' % f)

                data = physion.analysis.read_NWB.Data(f, verbose=False)
                nROIs_original = data.nROIs # manually-selected ROIs

                data.build_dFoF(**dFoF_parameters)
                data.build_Deconvolved(Tau=TAU_DECONVOLUTION)

                nROIs_final = data.nROIs # ROIs after dFoF criterion

                protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0]
                Episodes = physion.analysis.episodes.build.EpisodeData(data, 
                                                        quantities=['Deconvolved'], 
                                                        protocol_name=protocol_name, 
                                                        verbose=False)

                epCond = Episodes.find_episode_cond(key=['contrast', 'angle'], 
                                                        value=[contrast, orientation])

                significant = np.zeros(data.nROIs, dtype=bool)
                for i in range(data.nROIs):
                        cell_resp = Episodes.compute_summary_data(stat_test_props,
                                                                response_args=dict(roiIndex=i))
                        cond = (cell_resp['angle']==orientation) &\
                                        (cell_resp['contrast']==contrast)
                        significant[i] = cell_resp['significant'][cond][0]
                        
                Response = {
                      't':Episodes.t,
                      'Deconvolved':Episodes.Deconvolved[epCond,:,:].mean(axis=0),
                      'significant':significant,
                      'nROIs_original': nROIs_original,
                      'nROIs_final': nROIs_final,

                }

                Responses.append(Response)

        # saving data
        np.save(os.path.join(tempfile.tempdir, 
                        'Deconvolved_WT_angle-%.1f.npy' % angle),
                Responses)

# %%

# loading data
Responses = np.load(os.path.join(tempfile.tempdir, 
                                     'Deconvolved_WT_angle-0.0.npy'), 
                  allow_pickle=True)

responses = np.array([np.mean(S['Deconvolved'], axis=0) for S in Responses])

# Plot
fig, ax = pt.figure(ax_scale=(1.2, 1))

pt.plot(Responses[0]['t'], np.mean(responses, axis=0), 
        sy=stats.sem(responses, axis=0), ax=ax, ms=3)

pt.annotate(ax, 'N=%i sessions' % len(Responses), (0.8,1))

pt.set_plot(ax, 
            title='c=%.1f, $\\theta$=%.1f' % (contrast, orientation),
            ylabel='$\delta$ $\Delta$F/F',  
            xlabel='time (s)')

# %%

# %%
