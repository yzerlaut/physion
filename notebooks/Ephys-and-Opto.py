# %% [markdown]
# # Analysis of Eletrophysiology Data and Optogenetic Manipulation

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

sys.path += ['../src'] # add src code directory for physion
import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data


pt.set_style('dark')

datafile = os.path.join(os.path.expanduser('~'), 
                        'DATA', '2026_04_24', '2026_04_24-12-45-49.nwb')

data = Data(datafile)
data.build_pupil_diameter()

# data.build_running_speed()
# data.build_LFP(specific_time_sampling=data.t_running_speed)
# data.build_MUA(specific_time_sampling=data.t_running_speed)
# %%
data.build_suSpikes() # builds data.suSpikes
# data.build_suWaveforms() # builds data.suWaveforms
# data.read_optogen() # builds data.LED
# data.build_muEvents() # builds data.muEvents

# %%
print(data.protocols)

# %%

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
AX[2].plot(data.t_suSpikes[cond], data.suSpikes[:,cond].sum(axis=0))



# %%

from physion.analysis.episodes.build import EpisodeData

ep = EpisodeData(data, prestim_duration=2., 
                 quantities=['suSpikes', 'LED'],
                 protocol_name='ffDG-4dir-2ctrst+1sPrePostOpto')
# %%

for i range(10):
    plt.plot(ep.t, ep.LED[i, :])


# %%
from physion.utils import plot_tools as pt

LED_on = ep.LED.mean(axis=1)>0 # LED "On" episode condition

fig, ax = pt.figure()

ax.plot(ep.t, ep.suSpikes[LED_on, :,:].mean(axis=(0,1)))
ax.plot(ep.t, ep.suSpikes[~LED_on,:, :].mean(axis=(0,1)))

# %%
data.build_suWaveforms() # builds data.suWaveforms
from physion.dataviz.ephys import show_waveforms
for unit in range(10):
    fig, ax = show_waveforms(data, unit_id=unit)
    ax.set_title('unit #%i' % (unit+1))
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

# # %%
# import numpy as np
# df = '/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16/Record Node 101/experiment1/recording1/continuous/OneBox-100.ProbeA/kilosort4'
# data = {}
# for key in [f for f in os.listdir(df) if '.npy' in f]:
#     # data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
#     # exec('%s = np.load("%s/%s", allow_pickle=True)' % (key.replace('.npy',''), df, key))
#     data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
# # %%
# templates = np.load(os.path.join(df, 'templates.npy'), allow_pickle=True)

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
# sample_index = np.array([peaks.item(i)[0] for i in range(len(peaks))])
# channel = np.array([peaks.item(i)[1] for i in range(len(peaks))])
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
