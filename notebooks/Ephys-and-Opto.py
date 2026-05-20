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
data.build_running_speed()
data.build_LFP(specific_time_sampling=data.t_running_speed)
data.build_MUA(specific_time_sampling=data.t_running_speed)
# %%
from physion.analysis.episodes.build import EpisodeData
ep = EpisodeData(data, quantities=['LFP']) 
print(ep.LFP.shape)
# %
from physion.dataviz.episodes.trial_average import plot
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %%
import numpy as np
df = '/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16/Record Node 101/experiment1/recording1/continuous/OneBox-100.ProbeA/kilosort4'
data = {}
for key in [f for f in os.listdir(df) if '.npy' in f]:
    # data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
    # exec('%s = np.load("%s/%s", allow_pickle=True)' % (key.replace('.npy',''), df, key))
    data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
# %%
templates = np.load(os.path.join(df, 'templates.npy'), allow_pickle=True)

# %%
import matplotlib.pylab as plt
n=1
x, y = data['spike_positions'][n]


template_id = 2
n = find_center_channel(template_id, data)
for i in range(400):
    print(find_center_channel(i, data))

# %%
template_id = 10

def find_center_channel(template_id, data):
    return np.argmax(np.std(data['templates'][template_id,:,:],axis=0))

def plot_template(data, 
                  template_id,
                  axes = (2,9)):

    n = find_center_channel(template_id, data)
    fig, AX = pt.figure(axes=axes, ax_scale=(.6,.4), wspace=0.3, hspace=0.1)
    for i, ii in enumerate(np.arange(2)):
        for j, jj in enumerate(np.arange(-3,5)):
            AX[j+1][i].axis('off')
            if (n+2*jj+i)>0:
                AX[axes[1]-j-1][axes[0]-i-1].plot(data['templates'][template_id,:,n+2*jj+ii])
                pt.annotate(AX[axes[1]-j-1][axes[0]-i-1], 'ch.%i' % (n+2*jj+ii), (1,1), ha='right', fontsize=4)

    pt.annotate(AX[0][1], '#%i' % template_id, (1,1), va='top', ha='right', fontsize=6)

    AX[0][0].axis('off')
    AX[0][1].axis('off')
    pt.draw_bar_scales(AX[0][0], 
                    Xbar=30, Xbar_label='1ms',
                    Ybar=2, Ybar_label='2$\mu$V')
    pt.set_common_ylims(AX)
    pt.set_common_xlims(AX)

plot_template(data, 70)

# %%
import spikeinterface.full as si
# %%
df = '/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16/Record Node 101/experiment1/recording1/continuous/OneBox-100.ProbeA/kilosort4'
sorting = si.read_kilosort(df)
# %%
rec = si.read_openephys('/Users/yann/DATA/2026_04_24/2026-04-24_12-23-16',
                    stream_name='Record Node 101#OneBox-100.ProbeA')
analyzer = si.create_sorting_analyzer(sorting=sorting, 
                                    recording=rec)
analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels"])

# %%
templates.shape
# %%
# should be: (61, 384, 409)
features=np.array([
    [templates[:,i,k] for k in range(templates.shape[2])]\
        for i in range(templates.shape[1])])
features.shape

# %%
traces = rec.get_traces()

# %%
nMax = rec.get_num_frames()
downsampling_factor=25
new = traces[:downsampling_factor*int(nMax/downsampling_factor),:]
new = np.abs(new)
new = new.reshape((int(nMax/downsampling_factor),25,384))
new = new.mean(axis=1)
print(new.shape)
# %%
downsampling_factor=25
nMax = rec.get_num_frames()
new = np.abs(rec.get_traces()[:downsampling_factor*int(nMax/downsampling_factor),:]).reshape((int(nMax/downsampling_factor),25,384)).mean(axis=1)
print(new.shape)

# %%
