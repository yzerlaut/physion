
# %% [markdown]
#
# # Analysis of Neuropixels data
#

# %%
import sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
import numpy as np
import pandas as pd

from open_ephys.analysis import Session as OpenEphysSession
from physion.assembling.dataset import read_spreadsheet

import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_04_10').replace('/', os.path.sep)

# datatable, _, analysis = read_spreadsheet(\
#                         os.path.join(datafolder, 'DataTable0.xlsx'),
#                                    get_metadata_from='files')

# INTERPROTOCOL_WINDOW = 10. # 
# PROBE_NAME = 'ProbeA'
# EXP = 1 # 
# NODE=0

class Data:

    def __init__(self, datafolder, iRec):

        datatable, _, _ = read_spreadsheet(\
                                os.path.join(datafolder, 'DataTable0.xlsx'),
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
        rec = session.recordnodes[node].recordings[rec_id]

        self.LFP = rec.continuous['ProbeA'].samples[nStart:nStop,:]

        # @Sally
        # self.spikes = ...

# %%
data = Data(datafolder, 0)

t0, length = 0, 60
fig, AX = pt.figure(axes_extents=[[[1,3]],[[1,1]]], ax_scale=(3,1))

SHIFT = 1000 # 1mV between each channel
cond = (data.t_probe>t0) & (data.t_probe<(t0+length))

for chan in range(10):
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

channel = 100
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

# %%
# CSD analysis
csd_events = []
window = [-0.4,2] # temporal window
for e in events:
    cond = (data.t_probe>(e+window[0])) & (data.t_probe<(e+window[1]))
    # lfp = gaussian_filter1d(data.LFP[cond,:].mean(axis=-1), 500)
    lfp = gaussian_filter1d(data.LFP[cond,0], 500)
    pre = (data.t_probe[cond]>(e+window[0])) & (data.t_probe[cond]<e)
    lfp_events.append(lfp-lfp[pre].mean())
minLength = min([len(l) for l in lfp_events])
lfp_events = [l[:minLength] for l in lfp_events]
t = data.t_probe[:minLength]+window[0]

fig, ax = pt.figure(ax_scale=(2,3))
pt.plot(t, 1e-3*np.mean(lfp_events, axis=0), sy=1e-3*np.std(lfp_events, axis=0), ax=ax)
pt.set_plot(ax, xlabel='time from stim. (s)', ylabel='LFP (mV)')

# %%
