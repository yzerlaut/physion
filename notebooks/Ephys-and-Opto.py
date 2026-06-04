# %% [markdown]
# # Analysis of Electrophysiology Data and Optogenetic Manipulation

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['../src'] # add src code directory for physion
import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles

dataset = scan_folder_for_NWBfiles(\
        os.path.join(os.path.expanduser('~'), 
            'DATA', '2026_04_24'))
# %%
pt.set_style('dark')

data = Data(dataset['files'][1])
# data.build_pupil_diameter()
data.build_suSpikes() # builds data.suSpikes
data.build_suWaveforms() # builds data.suWaveforms

# data.build_running_speed()
# data.build_LFP(specific_time_sampling=data.t_running_speed)
# data.build_MUA(specific_time_sampling=data.t_running_speed)
# data.build_suWaveforms() # builds data.suWaveforms
# data.read_optogen() # builds data.LED
# data.build_muEvents() # builds data.muEvents

# %%
# print(data.protocols)

# %%
from scipy.ndimage import gaussian_filter1d

fig, AX = pt.figure(axes=(1,3), ax_scale=(2,1), hspace=0)

tmax = 160
# 1) LED
cond = (data.t_LED<tmax)
AX[1].plot(data.t_LED[cond], data.LED[cond])
# 2) Pupil
cond = (data.t_pupil[:]<tmax)
AX[0].plot(data.t_pupil[cond], data.pupil_diameter[cond])
# 2) Firing count across units
cond = (data.t_suSpikes<tmax)
firing = gaussian_filter1d(
           data.suSpikes[:,cond].sum(axis=0), 30)
AX[2].plot(data.t_suSpikes[cond], firing) 



# %%

from physion.analysis.episodes.build import EpisodeData

ep = EpisodeData(data, prestim_duration=4., 
                 quantities=['suSpikes', 'LED'],
                 protocol_name='ffDG-4dir-2ctrst+1sPrePostOpto')


# %%
from physion.utils import plot_tools as pt
from physion.dataviz.ephys import show_waveforms

LED_on = ep.LED.mean(axis=1)>0 # LED "On" episode condition

# %%
import matplotlib.pylab as plt

def plot_unit(unit):
    fig = plt.figure(figsize=(4,2))
    axWF = fig.add_axes([0,0,0.4,1])
    axBlank = fig.add_axes([0.5,0.5,0.5,0.4])
    axLED = fig.add_axes([0.5,0,0.5,0.4])
    for ax, label, cond in zip([axBlank, axLED], 
                            ['blank', 'LED'],
                            [~LED_on, LED_on]):
        rate = ep.suSpikes[cond, unit, :].mean(axis=0)/(ep.t[1]-ep.t[0])
        # smoothing:
        rate = gaussian_filter1d(rate, 50)
        ax.fill_between(ep.t, 0*ep.t, rate)
        # ax.plot(ep.t, rate)
        pt.annotate(ax, label, (0.5,1), va='top', ha='center')
        pt.draw_bar_scales(ax, Xbar=1, Ybar=2,
                        Xbar_label='1s', Ybar_label='2Hz')
        ax.axis('off')
    pt.set_common_ylims([axBlank, axLED])
    for ax in [axBlank, axLED]:
        ax.fill_between([0,2], [0,0],
                    ax.get_ylim()[1]*np.ones(2), alpha=.1)
    axLED.fill_between(ep.t, 0*ep.t,
                ax.get_ylim()[1]*ep.LED[cond, :].mean(axis=0), alpha=0.1)
    _ = show_waveforms(data, unit_id=unit, ax=axWF,
                       channels_around=5)
    pt.annotate(fig, 'unit #%i' % (unit+1), 
                (1,1), va='top', ha='right')
    return fig, [axBlank, axLED], axWF
# %%
for unit in range(data.suSpikes.shape[0]):
    # ax.set_title('unit #%i' % (unit+1))
    # fig, ax = show_waveforms(data, unit_id=unit, ax=axWF)
    plot_unit(unit)
    
# 
# %%
for i, unit in enumerate(data.nwbfile.units):
    print(data.tlim[1], unit.spike_times.values[:][0][-1:])

# # %%
# from physion.analysis.episodes.build import EpisodeData
# ep = EpisodeData(data, quantities=['LFP']) 
# print(ep.LFP.shape)
# # %
# from physion.dataviz.episodes.trial_average import plot
# import physion.utils.plot_tools as pt
# pt.set_style('dark')

# %%
import numpy as np
import pandas as pd

import csv
df = '/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16/Record Node 101/experiment1/recording1/continuous/OneBox-100.ProbeA/kilosort4'
def read_kilosort(df):
    data = {}
    for key in [f for f in os.listdir(df) if '.npy' in f]:
        # data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
        # exec('%s = np.load("%s/%s", allow_pickle=True)' % (key.replace('.npy',''), df, key))
        data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
    for key in [f for f in os.listdir(df) if '.tsv' in f]:
        rd = pd.read_csv(open(os.path.join(df, key)), sep = '\t')
        keys = list(rd.keys())
        for k in keys:
            if k!='cluster_id':
                data[key.replace('.tsv','')+'_'+k] = rd[k]
    return data
data = read_kilosort(df)
# %%
templates = np.load(os.path.join(df, 'templates.npy'), allow_pickle=True)

# # %%
# import matplotlib.pylab as plt
# n=1
# x, y = data['spike_positions'][n]


# template_id = 2
# n = find_center_channel(template_id, data)
# for i in range(400):
#     print(find_center_channel(i, data))

# # %%
# template_id = 10

# def find_center_channel(template_id, data):
#     return np.argmax(np.std(data['templates'][template_id,:,:],axis=0))

# def plot_template(data, 
#                   template_id,
#                   axes = (2,9)):

#     n = find_center_channel(template_id, data)
#     fig, AX = pt.figure(axes=axes, ax_scale=(.6,.4), wspace=0.3, hspace=0.1)
#     for i, ii in enumerate(np.arange(2)):
#         for j, jj in enumerate(np.arange(-3,5)):
#             AX[j+1][i].axis('off')
#             if (n+2*jj+i)>0:
#                 AX[axes[1]-j-1][axes[0]-i-1].plot(data['templates'][template_id,:,n+2*jj+ii])
#                 pt.annotate(AX[axes[1]-j-1][axes[0]-i-1], 'ch.%i' % (n+2*jj+ii), (1,1), ha='right', fontsize=4)

#     pt.annotate(AX[0][1], '#%i' % template_id, (1,1), va='top', ha='right', fontsize=6)

#     AX[0][0].axis('off')
#     AX[0][1].axis('off')
#     pt.draw_bar_scales(AX[0][0], 
#                     Xbar=30, Xbar_label='1ms',
#                     Ybar=2, Ybar_label='2$\mu$V')
#     pt.set_common_ylims(AX)
#     pt.set_common_xlims(AX)

# plot_template(data, 70)

# # %%
# %%
# import spikeinterface.full as si
# rec = si.read_openephys('/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16',
#                     stream_name='Record Node 101#OneBox-100.ProbeA')

# df = '/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16/Record Node 101/experiment1/recording1/continuous/OneBox-100.ProbeA/kilosort4'
# sorting = si.read_kilosort(df)

# %%


# %%
# job_kwargs = dict(n_jobs=-1, progress_bar=True, chunk_duration="1s")

# make the SortingAnalyzer with necessary and some optional extensions
# sorting_analyzer = si.create_sorting_analyzer(sorting, rec,
#                                               format="binary_folder", folder="/my_sorting_analyzer",
#                                               **job_kwargs)

# # %%
# import spikeinterface.full as si
# rec = si.read_openephys('/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16',
#                     stream_name='Record Node 101#OneBox-100.ProbeA')
# rec = rec.select_channels(rec.get_channel_ids()[::20])
# # %%
# analyzer = si.create_sorting_analyzer(sorting=sorting, 
#                                     recording=rec)
# analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])

# # %%
# templates = np.load(os.path.join(df, 'templates.npy'), 
#                             allow_pickle=True)
# templates.shape
# # %%
# # should be: (61, 384, 409)
# features=np.array([
#     [templates[:,i,k] for k in range(templates.shape[2])]\
#         for i in range(templates.shape[1])])
# features.shape

# # %%

# # %%
# traces = rec.get_traces()

# # %%
# nMax = rec.get_num_frames()
# downsampling_factor=25
# new = traces[:downsampling_factor*int(nMax/downsampling_factor),:]
# new = np.abs(new)
# new = new.reshape((int(nMax/downsampling_factor),25,384))
# new = new.mean(axis=1)
# print(new.shape)
# # %%
# downsampling_factor=25
# nMax = rec.get_num_frames()
# new = np.abs(rec.get_traces()[:downsampling_factor*int(nMax/downsampling_factor),:]).reshape((int(nMax/downsampling_factor),25,384)).mean(axis=1)
# print(new.shape)

# # %%
# # %%
# import spikeinterface.full as si
# rec = si.read_openephys('/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16',
#                     stream_name='Record Node 101#OneBox-100.ProbeA')
# # rec = rec.select_channels(rec.get_channel_ids()[::20])
# rec = rec.frame_slice(1e8,1.1e8)
# # %%
# from spikeinterface.sortingcomponents import peak_detection
# peaks = peak_detection.detect_peaks(rec)
# # %%
# sample_indices = np.array([peaks.item(i)[0] for i in range(len(peaks))])
# channels = np.array([peaks.item(i)[1] for i in range(len(peaks))])
# sample_index
# # %%
# np.unique([peaks.item(i)[1] for i in range(len(peaks))])

# # %%
# data = np.array([peaks.item(i)[:2] for i in range(len(peaks))])
# data

# # %%
# bad_chans = si.detect_bad_channels(rec)
# # %%
# rec = rec.remove_channels(bad_chans[0])
# # %%

# %%
