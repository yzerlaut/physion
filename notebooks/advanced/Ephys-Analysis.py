
# %% [markdown]
#
# # Analysis of Neuropixels data
#

# %%
import os, sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
import numpy as np
import pandas as pd

from open_ephys.analysis import Session as OpenEphysSession
from physion.assembling.dataset import read_spreadsheet

import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_06_09').replace('/', os.path.sep)

# datatable, _, analysis = read_spreadsheet(\
#                         os.path.join(datafolder, 'DataTable.xlsx'),
#                                    get_metadata_from='files')

# INTERPROTOCOL_WINDOW = 10. # 
# PROBE_NAME = 'ProbeA'
# EXP = 1 # 
# NODE=0

class Data:

    def __init__(self, datafolder, iRec):

        datatable, _, _ = read_spreadsheet(\
                                os.path.join(datafolder, 'DataTable.xlsx'),
                                        get_metadata_from='files')

        nidaq = np.load(os.path.join(datafolder, datatable['time'][iRec], 'Nidaq.npy'),
                        allow_pickle=True).item()
        self.t_nidaq = np.arange(0, len(nidaq['digital'][0]))*nidaq['dt']
        self.visStim = nidaq['digital'][3]

        nStart, nStop = datatable['nStart'][iRec], datatable['nStop'][iRec]
        self.t_probe = np.linspace(0, self.t_nidaq[-1], nStop-nStart)

        # load the open-ephys data:
        session = OpenEphysSession(\
            os.path.join(datafolder, datatable['Npx-Folder'][iRec]))

        node = int(datatable['Npx-Rec'][iRec].split('node')[1].split('/')[0])
        rec_id = int(datatable['Npx-Rec'][iRec].split('rec')[1])-1
        print(node, rec_id)
        rec = session.recordnodes[node].recordings[rec_id]

        self.LFP = rec.continuous['ProbeA'].samples[nStart:nStop,:]

        # @Sally
        # self.spikes = ...

# %%
data = Data(datafolder, 1)

t0, length = 0, 60
fig, AX = pt.figure(axes_extents=[[[1,3]],[[1,1]]], ax_scale=(3,1))

SHIFT = 1500 # 1mV between each channel
cond = (data.t_probe>t0) & (data.t_probe<(t0+length))

for chan in np.arange(15)*10:
    lfp = data.LFP[cond,chan]
    lfp = lfp-lfp.mean()
    AX[0].plot(data.t_probe[cond], lfp+chan*SHIFT, lw=0.5, color=pt.plt.cm.tab20(chan))
pt.set_plot(AX[0], ['bottom'], ylabel='LFP')
pt.draw_bar_scales(AX[0], Xbar=1e-3, Ybar=2000, Ybar_label='2mv')

cond = (data.t_nidaq>t0) & (data.t_nidaq<(t0+length))
AX[1].plot(data.t_nidaq[cond], data.visStim[cond])
pt.set_plot(AX[1], ['bottom'], xlabel='time (s)', ylabel='vis. stim.\n onset')

# %%
from scipy.ndimage import gaussian_filter1d
events = data.t_nidaq[np.flatnonzero(data.visStim[1:]>data.visStim[:-1])]

channel = 154
lfp_events = []
window = [-0.4,2] # temporal window
for e in events:
    cond = (data.t_probe>(e+window[0])) & (data.t_probe<(e+window[1]))
    # lfp = gaussian_filter1d(data.LFP[cond,:].mean(axis=-1), 500)
    lfp = gaussian_filter1d(data.LFP[cond,channel], 500)
    pre = (data.t_probe[cond]>(e+window[0])) & (data.t_probe[cond]<e)
    lfp_events.append(lfp-lfp[pre].mean())
minLength = min([len(l) for l in lfp_events])
lfp_events = [l[:minLength] for l in lfp_events]
t = data.t_probe[:minLength]+window[0]

fig, ax = pt.figure(ax_scale=(2,3))
pt.plot(t, 1e-3*np.mean(lfp_events, axis=0), sy=1e-3*np.std(lfp_events, axis=0), ax=ax)
pt.set_plot(ax, xlabel='time from stim. (s)', ylabel='LFP (mV)')

# %% [markdown]
# # CSD analysis

# %%
datafolder = os.path.expanduser('~/DATA/2026_04_24/2026-04-24_12-23-16')
session = OpenEphysSession(datafolder)
rec = session.recordnodes[0].recordings[0]


# %%
probes = [cRec['folder_name'] for cRec in rec.info['continuous']\
                if 'Probe' in cRec['folder_name']]
print(probes)
# %%
len(rec.info['continuous'][0]['channels'])
# %%
probes = [p for p in rec.continuous.keys() if (type(p)==str) and ('Probe' in p)]
probes
# %%
from spikeinterface.extractors import read_openephys   # binary or classic
 
# ── ProbeInterface (Neuropixels channel geometry) ─────────────────────────────
from probeinterface import get_probe
 

# %%
# get_probe('IMEC', 'Neuropixels2.0')
# %%
import probeinterface 
probeinterface.list_all_probes()
# %%
from probeinterface import Probe, get_probe
from probeinterface.plotting import plot_probe
probe = get_probe('imec', 'NP2004') # single shank probe
plot_probe(probe)
# %%
from spikeinterface.extractors import read_openephys   # binary or classic
# %%
siRec = read_openephys(datafolder,
                       stream_name='Record Node 101#OneBox-100.ProbeA')
# %%
from spikeinterface import extractors 

# %%
stream_name='{recorded_processor} {recorded_processor_id}'.format(\
    **rec.info['continuous'][1])
stream_name
siRec = extractors.read_openephys(datafolder, stream_id='1')
# %%
p= siRec.get_annotation('probes_info')[0]
probe = get_probe("imec", p["model_name"])
probe.set_device_channel_indices(np.arange(siRec.get_num_channels()))
siRec = siRec.set_probe(siRec.get_probe())#, group_mode="by_shank")
# %%
probes = siRec.get_annotation('probes_info')

# %%
import spikeinterface.full as si
rec_lfp = si.resample(siRec, resample_rate=1250)

# %%
rec_lfp.get_num_frames()
# %%


# %%
#####################################################################################
###        data analysis   ##########################################################
#####################################################################################

from scipy.ndimage import gaussian_filter1d

datatable, _, _ = read_spreadsheet(\
            os.path.join(datafolder, 'DataTable.xlsx'),
                    get_metadata_from='files')

def get_channel_subsampled_LFP(rec, iRange, 
                               channel_subsampling,
                               temporal_smoothing=10):

    LFP, chan = [], 0

    while chan<(rec.continuous['ProbeA'].samples.shape[1]-channel_subsampling):

        lfp = []
        for c in range(channel_subsampling):
            lfp.append(\
                rec.continuous['ProbeA'].samples[iRange,chan+c])
        LFP.append(\
            gaussian_filter1d(\
                np.mean(lfp, axis=0), 
                    temporal_smoothing))

        chan += channel_subsampling
        
    return np.array(LFP)

def load_LFP_resp(datafolder, iRec,
                  channel_subsampling=4,
                  temporal_smoothing=10,
                  pre_window=0.5,
                  post_window=1):

    datatable, _, _ = read_spreadsheet(\
                            os.path.join(datafolder, 'DataTable.xlsx'),
                                    get_metadata_from='files')

    nidaq = np.load(os.path.join(datafolder, datatable['time'][iRec], 'Nidaq.npy'),
                    allow_pickle=True).item()
    t_nidaq = np.arange(0, len(nidaq['digital'][0]))*nidaq['dt']
    visStim = nidaq['digital'][3]
    
    nStart, nStop = datatable['nStart'][iRec], datatable['nStop'][iRec]
    t_probe = np.linspace(0, t_nidaq[-1], nStop-nStart)
    dt_probe = t_nidaq[-1]/(nStop-nStart)
    iPre, iPost = int(pre_window/dt_probe), int(post_window/dt_probe)
    # load the open-ephys data:
    session = OpenEphysSession(\
        os.path.join(datafolder, datatable['Npx-Folder'][iRec]))

    node = int(datatable['Npx-Rec'][iRec].split('node')[1].split('/')[0])
    rec_id = int(datatable['Npx-Rec'][iRec].split('rec')[1])-1
    rec = session.recordnodes[node].recordings[rec_id]

    # loop over stim events
    LFP_resp = []
    for i in np.flatnonzero(visStim[1:]>visStim[:-1]):
        iP = np.argmin((t_probe-t_nidaq[i])**2) # 
        # print(t_probe[iP])
        LFP_resp.append(\
            get_channel_subsampled_LFP(rec, iP+np.arange(-iPre, iPost),
                               channel_subsampling,
                               temporal_smoothing=temporal_smoothing))

    return t_probe[:iPost+iPre]-t_probe[iPre], np.array(LFP_resp)


def show_LFP(t, LFP):
    fig, ax = pt.figure(ax_scale=(2,3))
    shift= 10
    for i, c in enumerate(np.arange(LFP.shape[0])):
        lfp = LFP[c,:].mean(axis=0) # trial-average here
        ax.plot(t, lfp-np.nanmean(lfp[t<0])+c*shift, color=pt.viridis(i/LFP.shape[0]), lw=0.5)
    pt.set_plot(ax, xlabel='time (s)', 
                ylim=[-2000,200], 
                ylabel='uV')
    return fig, ax

iRecs = np.flatnonzero(datatable['protocol']=='flashed-stimuli')

for iRec in iRecs:

    t, LFP = load_LFP_resp(datafolder, iRec, 
                           channel_subsampling=4,
                           temporal_smoothing=100)
    show_LFP(t, LFP)

# %%
# %%
def show_CSD(t, LFP):
    CSD = np.diff(LFP, axis=0).T
    fig, ax = pt.figure(ax_scale=(2,3))
    ax.imshow(CSD, vmin=-1000, vmax=1000, cmap=pt.PiYG, aspect='auto',
            extent=(t[0], 0, t[-1]-t[0], CSD.shape[1]),
            origin='lower')
    pt.set_plot(ax, xlabel='time (s)', ylabel='channel (subsampled)')
    return fig, ax

iRec = 4 #flash

t, LFP = load_LFP_resp(datafolder, iRec, channel_subsampling=4,
                        temporal_smoothing=100)

show_CSD(t, LFP)

# %%

# %%
