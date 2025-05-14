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
# # Behavioral modulation of visually Evoked Activity

# %%
import sys, os
import numpy as np
from scipy.optimize import minimize

sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src')) # update to your "physion" location
import physion
import physion.utils.plot_tools as pt

# %% [markdown]
# ## Load the Episodes of a given Protocol in a Datafile 

# %%
filename = os.path.join(os.path.expanduser('~'), 
                        'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs',
                        '2023_02_15-13-30-47.nwb')

data = physion.analysis.read_NWB.Data(filename, verbose=False)
data.build_dFoF(method_for_F0='sliding_percentile', percentile=10., verbose=False)

Episodes = physion.analysis.process_NWB.EpisodeData(data,
                                                    quantities=['dFoF', 'running_speed', 'pupil_diameter'],
                                                    protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                                                    verbose=False,
                                                    dt_sampling=10)

# %%
# visualize those data
t0 = Episodes.time_start_realigned[0]-1
figRaw, _ = physion.dataviz.raw.plot(data, tlim=[t0,t0+300],
                                     settings=physion.dataviz.raw.find_default_plot_settings(data, with_subsampling=True))

# %% [markdown]
# ## Split Episodes according to Behavioral States

# %%
# Compute average behavior within episodes
withinEpisode_cond = (Episodes.t>0) & (Episodes.t<Episodes.time_duration[0])

Ep_run_speed = Episodes.running_speed[:,withinEpisode_cond].mean(axis=1)
Ep_pupil_size = Episodes.pupil_diameter[:,withinEpisode_cond].mean(axis=1)

# binning the data according to pupil level for analysis:
pupil_bins = np.linspace(Ep_pupil_size.min(), Ep_pupil_size.max(), 15)

bins = np.digitize(Ep_pupil_size, pupil_bins)
speed_binned, sb_std = np.zeros(len(pupil_bins)), np.zeros(len(pupil_bins))

for b in np.unique(bins):
    speed_binned[b-1] = np.mean(Ep_run_speed[bins==b])
    sb_std[b-1] = np.std(Ep_run_speed[bins==b])

# Run vs Rest --> speed threshold

speed_threshold = 0.1

# Constricted vs Dilated --> pupil threshold

def func(t, X):
    """ threshold-linear function """
    return np.array([X[1]*(tt-X[0]) if tt>X[0] else 0 for tt in t])
    
def to_minimize(X):
    return np.sum((speed_binned-func(pupil_bins, X))**2)
    
res = minimize(to_minimize,
               [pupil_bins.mean(), 1])

pupil_threshold = res.x[0]

# %%
# plot analysis quantities

fig, AX = pt.figure(axes=(3,1), figsize=(1.2,1.2))

pt.scatter(Ep_pupil_size, Ep_run_speed, ax=AX[0])

AX[1].plot([Ep_pupil_size.min(), Ep_pupil_size.max()], [speed_threshold,speed_threshold], 'r:')
pt.annotate(AX[1], 'speed \nthresh. ', (Ep_pupil_size.max(), speed_threshold),
            color='r', xycoords='data', fontsize=7)

pt.scatter(pupil_bins, speed_binned, sy=sb_std, ax=AX[1], ms=2, lw=0)

pt.plot(pupil_bins, speed_binned, sy=sb_std, ax=AX[2], ms=2, lw=1)
AX[2].plot(pupil_bins, func(pupil_bins, res.x), 'r-', lw=2)


AX[2].plot([pupil_threshold,pupil_threshold], [0, speed_binned[-1]], 'r:')
pt.annotate(AX[2], 'pupil \nthresh. ', (pupil_threshold, speed_binned[-1]),
            va='top', ha='right', color='r', xycoords='data', fontsize=7)

for ax, title in zip(AX, ['single episodes', 'binned ', 'thresh.-linear fit']):
    pt.set_plot(ax, xlabel='pupil size (mm)', ylabel=' run. speed (cm/s)      ', title=title)

# %%
# plot analysis output

fig, AX = pt.figure(axes=(3,1), figsize=(1.2,1.2))

run = Ep_run_speed>speed_threshold
pt.scatter(Ep_pupil_size[run], Ep_run_speed[run], color='tab:orange', ax=AX[0])
pt.scatter(Ep_pupil_size[~run], Ep_run_speed[~run], color='tab:blue', ax=AX[0])
pt.annotate(AX[0], '%i ep.' % np.sum(~run), (0,1), va='top', color='tab:blue')
pt.annotate(AX[0], '\n%i ep.' % np.sum(run), (0,1), va='top', color='tab:orange')

dilated = Ep_pupil_size>pupil_threshold
pt.scatter(Ep_pupil_size[dilated], Ep_run_speed[dilated], color='tab:orange', ax=AX[1])
pt.scatter(Ep_pupil_size[~dilated], Ep_run_speed[~dilated], color='tab:blue', ax=AX[1])
pt.annotate(AX[1], '%i ep.' % np.sum(~dilated), (0,1), va='top', color='tab:blue')
pt.annotate(AX[1], '\n%i ep.' % np.sum(dilated), (0,1), va='top', color='tab:orange')

for i, cond, title, color in zip(range(4),
                          [dilated & run, dilated & ~run, ~dilated & ~run, ~dilated & run],
                          ['dilated & run ', 'dilated & rest', 'constr. & rest', 'constr. & run '],
                          ['tab:orange', 'tab:green', 'tab:blue', 'r']):
    pt.scatter(Ep_pupil_size[cond], Ep_run_speed[cond], color=color, ax=AX[2])
    pt.annotate(AX[2], i*'\n'+title+': %i ep.' % np.sum(cond), (1,1), va='top', color=color)

for ax, title in zip(AX, ['rest / run', 'constricted / dilated', 'mixed states']):
    pt.set_plot(ax, xlabel='pupil size (mm)', ylabel=' run. speed (cm/s)      ', title=title)

# %%
# show running speed and pupil size in different states

fig, AX = pt.figure(axes=(3,2), figsize=(1.2,1.2), hspace=0.5)

for cond, color in zip([run, ~run], ['tab:orange', 'tab:blue']):
    pt.plot(Episodes.t, Episodes.running_speed[cond,:].mean(axis=0), 
            sy=Episodes.running_speed[cond,:].std(axis=0), color=color, ax=AX[0][0])
    pt.plot(Episodes.t, Episodes.pupil_diameter[cond,:].mean(axis=0), 
            sy=Episodes.pupil_diameter[cond,:].std(axis=0), color=color, ax=AX[1][0])
    
for cond, color in zip([dilated, ~dilated], ['tab:orange', 'tab:blue']):
    pt.plot(Episodes.t, Episodes.running_speed[cond,:].mean(axis=0), 
            sy=Episodes.running_speed[cond,:].std(axis=0), color=color, ax=AX[0][1])
    pt.plot(Episodes.t, Episodes.pupil_diameter[cond,:].mean(axis=0), 
            sy=Episodes.pupil_diameter[cond,:].std(axis=0), color=color, ax=AX[1][1])

for i, cond, color in zip(range(4),
                          [dilated & run, dilated & ~run, ~dilated & ~run, ~dilated & run],
                          ['tab:orange', 'tab:green', 'tab:blue', 'r']):
    pt.plot(Episodes.t, Episodes.running_speed[cond,:].mean(axis=0), 
            sy=Episodes.running_speed[cond,:].std(axis=0), color=color, ax=AX[0][2])
    pt.plot(Episodes.t, Episodes.pupil_diameter[cond,:].mean(axis=0), 
            sy=Episodes.pupil_diameter[cond,:].std(axis=0), color=color, ax=AX[1][2])
    
pt.set_common_ylims(AX[0])
pt.set_common_ylims(AX[1])

for i, label, Ax in zip(range(2), ['run. speed\n(cm/s)', 'pupil diam.\n(mm)'], AX):
    for j, title, ax in zip(range(3), ['rest / run', 'constricted / dilated', 'mixed states'], Ax):
        pt.set_plot(ax, ylabel=label if j==0 else '', xlabel='time from stim. (s)' if i==1 else '',
                    title=title if i==0 else '')
    

# %%
# show state-dependent evoked activity for all ROIs

fig, AX = pt.figure(axes=(3,data.nROIs), figsize=(1,1), hspace=0.3, wspace=0.8)
from scipy import stats

for roi in range(data.nROIs):

    for cond, color in zip([run, ~run], ['tab:orange', 'tab:blue']):
        pt.plot(Episodes.t, Episodes.dFoF[cond,roi,:].mean(axis=0), 
                sy=stats.sem(Episodes.dFoF[cond,roi,:], axis=0), color=color, ax=AX[roi][0])
        
    for cond, color in zip([dilated, ~dilated], ['tab:orange', 'tab:blue']):
        pt.plot(Episodes.t, Episodes.dFoF[cond,roi,:].mean(axis=0), 
                sy=stats.sem(Episodes.dFoF[cond,roi,:], axis=0), color=color, ax=AX[roi][1])
    
    for i, cond, color in zip(range(4),
                              [dilated & run, dilated & ~run, ~dilated & ~run],
                              ['tab:orange', 'tab:green', 'tab:blue']):
        pt.plot(Episodes.t, Episodes.dFoF[cond,roi,:].mean(axis=0), 
                sy=stats.sem(Episodes.dFoF[cond,roi,:], axis=0), color=color, ax=AX[roi][2])
        
    pt.set_common_ylims(AX[roi])
    pt.annotate(AX[roi][0], 'roi #%i' % (roi+1), (0,1), va='top', fontsize=7)
    for j, title in enumerate(['rest / run', 'constricted / dilated', 'mixed states']):
        pt.set_plot(AX[roi][j], ylabel='$\delta$ $\Delta$F/F' if j==0 else '', 
                    xticks_labels=None if roi==(data.nROIs-1) else [],
                    xlabel='time from stim. (s)' if roi==(data.nROIs-1) else '',
                    title=title if roi==0 else '')


# %% [markdown]
# ## Session Summary

# %%
fig, AX = pt.figure(axes=(3, 1), figsize=(1,1), hspace=0.3, wspace=0.8)
from scipy import stats


for cond, color in zip([run, ~run], ['tab:orange', 'tab:blue']):
    pt.plot(Episodes.t, Episodes.dFoF[cond,:,:].mean(axis=(0,1)), 
            sy=stats.sem(Episodes.dFoF[cond,:,:].mean(axis=1), axis=0), color=color, ax=AX[0])
    
for cond, color in zip([dilated, ~dilated], ['tab:orange', 'tab:blue']):
    pt.plot(Episodes.t, Episodes.dFoF[cond,:,:].mean(axis=(0,1)), 
            sy=stats.sem(Episodes.dFoF[cond,:,:].mean(axis=1), axis=0), color=color, ax=AX[1])

for i, cond, color in zip(range(4),
                          [dilated & run, dilated & ~run, ~dilated & ~run],
                          ['tab:orange', 'tab:green', 'tab:blue']):
    pt.plot(Episodes.t, Episodes.dFoF[cond,:,:].mean(axis=(0,1)), 
            sy=stats.sem(Episodes.dFoF[cond,:,:].mean(axis=1), axis=0), color=color, ax=AX[2])
    
pt.set_common_ylims(AX)
pt.annotate(AX[0], '%i ROIs\n' % data.nROIs, (0,1), ha='right')
for j, title in enumerate(['rest / run', 'constricted / dilated', 'mixed states']):
    pt.set_plot(AX[j], ylabel='$\delta$ $\Delta$F/F' if j==0 else '', 
                xticks_labels=None if roi==(data.nROIs-1) else [],
                xlabel='time from stim. (s)' if roi==(data.nROIs-1) else '',
                title=title)


# %% [markdown]
# ## Multiple Sessions

# %%
DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 'DATA', 'physion_Demo-Datasets', 'SST-WT', 'NWBs'))

Responses = {'run':[], 'rest':[],
             'constricted':[], 'dilated':[],
             'low':[], 'mid':[], 'high':[]}

for f in DATASET['files']:
    
    print(' - analyzing file: %s  [...] ' % f)
    data = physion.analysis.read_NWB.Data(f, verbose=False)
    data.build_dFoF(method_for_F0='sliding_percentile', neuropil_correction_factor=0.7, percentile=10., verbose=False)

    Episodes = physion.analysis.process_NWB.EpisodeData(data, 
                                                    quantities=['dFoF', 'running_speed', 'pupil_diameter'],
                                                    protocol_name=[p for p in data.protocols if 'ff-gratings' in p][0],
                                                    verbose=False, prestim_duration=3,
                                                    dt_sampling=10)

    # Run vs Rest :
    withinEpisode_cond = (Episodes.t>0) & (Episodes.t<Episodes.time_duration[0])
    Ep_run_speed = Episodes.running_speed[:,withinEpisode_cond].mean(axis=1)
    run = Ep_run_speed>speed_threshold
    
    Responses['run'].append(Episodes.dFoF[run,:,:].mean(axis=(0,1)))
    Responses['rest'].append(Episodes.dFoF[~run,:,:].mean(axis=(0,1)))
Responses['t'] = Episodes.t

# %%
fig, AX = pt.figure(axes=(2,1))

baselineCond = (Episodes.t>-0.1) & (Episodes.t<0)
for cond, color in zip(['run', 'rest'], ['tab:orange', 'tab:blue']):

    pt.plot(Responses['t'], np.mean(Responses[cond], axis=0), 
            sy = stats.sem(Responses[cond], axis=0), color=color, no_set=True, ax=AX[0])

    Responses['baselineSubstr_%s' % cond] = [\
                r-r[baselineCond].min() for r in Responses[cond]]
    
    pt.plot(Responses['t'], np.mean(Responses['baselineSubstr_%s' % cond], axis=0), 
            sy = stats.sem(Responses['baselineSubstr_%s' % cond], axis=0), color=color, no_set=True, ax=AX[1])
    
pt.set_plot(AX[0], ylabel='$\delta$ $\Delta$F/F', 
            xlabel='time from stim. (s)' , xlim=[-1, Responses['t'][-1]],
            title='N=%i sessions' % len(Responses['run']))


from scipy.optimize import minimize
t_cond = Responses['t']>(-0.1)

gains = []

for i in range(len(Responses['run'])):
    
    def to_minimize(X):
        return np.sum((X[0]*Responses['baselineSubstr_rest'][i][t_cond]-
                       Responses['baselineSubstr_run'][i][t_cond])**2)
        
    res = minimize(to_minimize, [1.05])
    gains.append(res.x[0])
    
AX[1].plot(np.array(Responses['t'])[t_cond], 
           np.mean(gains)*np.mean(Responses['baselineSubstr_rest'], axis=0)[t_cond], 'k:', lw=0.5)

pt.set_plot(AX[1], ylabel='$\delta$ $\Delta$F/F', 
            xlabel='time from stim. (s)' , xlim=[-.1, Responses['t'][-1]],
            title='baseline substracted')

pt.annotate(AX[1], 
            'gain = %.2f $\pm$ %.2f' % (np.mean(gains), stats.sem(gains)),
            (1,.7), va='top')

# %%
