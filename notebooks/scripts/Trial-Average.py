# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules for scientific analysis
import sys, pathlib, os, itertools
import numpy as np

# add the python path:
sys.path.append('../../src')
from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data, scan_folder_for_NWBfiles
from physion.analysis.process_NWB import EpisodeData
from physion.dataviz.episodes.trial_average import plot_trial_average

# %%
# load a datafile
datafolder = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022')
filename = os.path.join(datafolder, '2022_12_14-13-27-41.nwb')
data = Data(filename,
            verbose=False)
data.build_dFoF(verbose=False)

# %% [markdown]
# ## Showing single ROI

# %%
roiIndex = 0
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

fig, AX = pt.plt.subplots(5, len(protocols),
                          figsize=(7,5))
pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

STAT_TEST = {}
for protocol in protocols:
    # a default stat test
    STAT_TEST[protocol] = dict(interval_pre=[-1,0],
                               interval_post=[1,2],
                               test='ttest',
                               positive=True)
STAT_TEST['looming-stim']['interval_post'] = [2, 3]
STAT_TEST['drifting-gratings']['interval_post'] = [1.5, 2.5]
STAT_TEST['moving-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['random-dots']['interval_post'] = [1.5, 2.5]
STAT_TEST['static-patch']['interval_post'] = [0.5, 1.5]

for p, protocol in enumerate(protocols):
    
    episodes = EpisodeData(data, 
                           quantities=['dFoF'],
                           protocol_name=protocol,
                           verbose=False)

    varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
    varied_values = [episodes.varied_parameters[k] for k in varied_keys]
    
    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')
    
    i=0
    for values in itertools.product(*varied_values):
        
        stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)
        plot_trial_average(episodes, roiIndex=roiIndex,
                           condition=stim_cond,
                           with_stat_test=True,
                           stat_test_props=STAT_TEST[protocol],
                           with_std_over_trials=True,
                           AX=[[AX[i][p]]])
        
        AX[i][p].annotate('%s=%s' % (varied_keys, values),
                          (0,-0.1), fontsize=4,
                          rotation=90, ha='right',
                          #va='top', ha='center',
                          xycoords='axes fraction')
        
        i+=1
        
AX[-1][0].annotate('single ROIs \n --> mean$\pm$s.d. over n=10 trials', (0, 0),
                  xycoords='axes fraction')

pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    if np.isfinite(ax.dataLim.x0):
        pt.draw_bar_scales(ax,
                           Xbar=1., Xbar_label='1s',
                           Ybar=1, Ybar_label='1$\Delta$F/F', fontsize=7)
pt.set_common_xlims(AX)
fig.savefig('/home/yann.zerlaut/Desktop/NDNF-summary/single-ROI.svg')

# %% [markdown]
# ## Showing average over ROIs

# %%
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

# prepare array for final results (averaged over sessions)
RESULTS = {}
for protocol in protocols:
    RESULTS[protocol] = {'significant':[], 'response':[], 'session':[]}

                               
fig, AX = pt.plt.subplots(5, len(protocols),
                          figsize=(7,5))
pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

for p, protocol in enumerate(protocols):
    
    episodes = EpisodeData(data, 
                           quantities=['dFoF'],
                           protocol_name=protocol,
                           verbose=False)
    varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
    varied_values = [episodes.varied_parameters[k] for k in varied_keys]
    
    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')
    
    i=0
    for values in itertools.product(*varied_values):
        
        stim_cond = episodes.find_episode_cond(key=varied_keys, value=values)
        plot_trial_average(episodes, 
                           condition=stim_cond,
                           with_stat_test=True,
                           stat_test_props=STAT_TEST[protocol],
                           with_std_over_rois=True,
                           AX=[[AX[i][p]]])
        
        AX[i][p].annotate('%s=%s' % (varied_keys, values),
                          (0,-0.1), fontsize=4,
                          rotation=90, ha='right',
                          #va='top', ha='center',
                          xycoords='axes fraction')
        
        RESULTS[protocol]['significant'].append([])
        RESULTS[protocol]['response'].append([])
        RESULTS[protocol]['session'].append([])
        i+=1
        
AX[-1][0].annotate('single session \n --> mean$\pm$s.d. over n=%i ROIs' % data.nROIs, (0, 0),
                  xycoords='axes fraction')

pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.axis('off')
    if np.isfinite(ax.dataLim.x0):
        pt.draw_bar_scales(ax,
                           Xbar=1., Xbar_label='1s',
                           Ybar=1, Ybar_label='1$\Delta$F/F', fontsize=7)
pt.set_common_xlims(AX)
fig.savefig('/home/yann.zerlaut/Desktop/NDNF-summary/single-session.svg')

# %% [markdown]
# ## Average over sessions

# %%
DATASET = scan_folder_for_NWBfiles(datafolder, verbose=False);

# %%
response_significance_threshold=0.01

# prepare array for final results (averaged over sessions)
RESULTS = {}
for protocol in protocols:
    RESULTS[protocol] = {'significant':[], 'response':[], 'session':[]}
protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol

# prepare array for final results (averaged over sessions)
RESULTS = {}
for p, protocol in enumerate(protocols):
    RESULTS[protocol] = {'significant':[], 'response':[], 'session':[]}
    episodes = EpisodeData(data, 
                           quantities=['dFoF'],
                           protocol_name=protocol,
                           verbose=False)
    varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
    varied_values = [episodes.varied_parameters[k] for k in varied_keys]
    for values in itertools.product(*varied_values):
        RESULTS[protocol]['significant'].append([])
        RESULTS[protocol]['response'].append([])
        RESULTS[protocol]['session'].append([])

for s, filename in enumerate(DATASET['files']):
    
    data = Data(filename,
                verbose=False)
    
    if data.metadata['protocol']=='NDNF-protocol':
        
        data.build_dFoF(verbose=False)
        
        for p, protocol in enumerate(protocols):
    
            episodes = EpisodeData(data, 
                                   quantities=['dFoF'],
                                   protocol_name=protocol,
                                   verbose=False)

            varied_keys = [k for k in episodes.varied_parameters.keys() if k!='repeat']
            varied_values = [episodes.varied_parameters[k] for k in varied_keys]
    
            for roi in range(data.nROIs):
                    
                summary = episodes.compute_summary_data(STAT_TEST[protocol],
                                              response_args={'quantity':'dFoF', 'roiIndex':roi},
                                              response_significance_threshold=response_significance_threshold)
                
                for v, values in enumerate(itertools.product(*varied_values)):

                    # build stim cond y iteration over values
                    stim_cond = np.ones(len(summary['value']), dtype=bool) # all true
                    for k, val in zip(varied_keys, values):
                        stim_cond = stim_cond & (summary[k]==val)
                
                    RESULTS[protocol]['response'][v].append(np.mean(summary['value'][stim_cond]))
                    RESULTS[protocol]['session'][v].append(s)
                    RESULTS[protocol]['significant'][v].append(bool(summary['significant'][stim_cond]))

# %%
                              
fig, AX = pt.plt.subplots(2, len(protocols),
                          figsize=(7,2))
pt.plt.subplots_adjust(wspace=0.5, hspace=0.1, top=.75, bottom=0)

for p, protocol in enumerate(protocols):
    
    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')

    frac_resp_per_session = []
    
    for v in range(len(RESULTS[protocol]['response'])):
        
        values_per_session = []
        
        for session in np.unique(RESULTS[protocol]['session'][v]):
        
            session_cond = np.array(RESULTS[protocol]['session'][v])==session
            values_per_session.append(np.array(RESULTS[protocol]['response'][v])[session_cond].mean())
            frac_resp_per_session.append(np.sum(np.array(RESULTS[protocol]['significant'][v])[session_cond])/\
                                         len(np.array(RESULTS[protocol]['significant'][v])[session_cond]))

        AX[1][p].bar([v], 
                     [np.mean(values_per_session)],
                     yerr=[np.std(values_per_session)])
    AX[0][p].axis('off')
    inset = pt.inset(AX[0][p], (0.1, 0.1, 0.8, 0.8))
    inset.annotate('%.1f$\pm$%.1f%%' % (100*np.mean(frac_resp_per_session),100*np.std(frac_resp_per_session)),
                       (0.5,1), xycoords='axes fraction', ha='center', fontsize=7)
    pt.pie([np.mean(frac_resp_per_session), 1-np.mean(frac_resp_per_session)], 
               ax=inset, COLORS=['tab:green', 'lightgrey'])
        
pt.set_common_ylims(AX)
for ax in pt.flatten(AX):
    ax.xaxis.set_visible(False)
pt.set_common_xlims(AX)
AX[0][0].annotate('  fraction   \n  responsive', (0,0.5), xycoords='axes fraction', rotation=90, 
                  ha='right', va='center')
AX[1][0].set_ylabel('mean $\Delta$F/F')
fig.savefig('/home/yann.zerlaut/Desktop/NDNF-summary/session-summary.svg')

# %%
