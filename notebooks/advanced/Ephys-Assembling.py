
# %% [markdown]
#
# # Assemble Neuropixels data
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```
# ## 1) 
# Run:
# ```
# python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA\2026_02_13
# ```
# this will create a file: `%USERPROFILE%\DATA\DataTable0.xlsx`  
#      move it to  ~/DATA/2026_02_13/DataTable0.xlsx
#
# Then fill its neuropixels folder (`Npx-Folder`) and recordings information (`Npx-Rec`).    
#
#       N.B. you can use the code below to guide filling the recordings info

# %%
import sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
import numpy as np
import pandas as pd

from open_ephys.analysis import Session

from physion.assembling.dataset import read_spreadsheet
from physion.acquisition.tools import find_line_props
import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_02_13').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
PROBE_NAME = 'ProbeA'
NODE = 0 # change if you have several record nodes and you want to consider another one
EXP = 1 # 


# %% [markdown]
#
# ## Load Table data

# %%
# datafolder = os.path.expanduser('~/DATA/2026_02_20').replace('/', os.path.sep)

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable0.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
def load_nidaq_synch_signal(folder):
    """ """
    with open(os.path.join(folder, 'metadata.json')) as f:
        metadata = json.load(f)
    NIdaq = np.load(os.path.join(folder, 'NIdaq.npy'),
                    allow_pickle=True).item()
    props = find_line_props(
                metadata['NIdaq']['digital-outputs']['line-labels'])
    ephysSynch_signal = NIdaq['digital'][props['chan']]
    t = np.arange(len(ephysSynch_signal))*NIdaq['dt']
    pulse_onsets = t[:-1][np.flatnonzero(ephysSynch_signal[1:]>ephysSynch_signal[:-1])]
    return t, ephysSynch_signal, pulse_onsets

DF = pd.DataFrame(columns=['time', 'Npx-Rec', 'daq-nEpisodes', 'ephys-nEpisodes', 'i0', 'i1', 'nStart', 'nStop'])
DF['time'] = datatable['time']

# loop over protocols
# print(' ==== PROTOCOLS FROM NIDAQ DATA ====  ')
for iRec, protocol in enumerate(datatable['protocol']):
    _, _, onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    # print(' rec #%i) n=%i episodes, %s' % (iRec+1, len(onsets), protocol))
    DF.loc[iRec, 'daq-nEpisodes'] = len(onsets)

# %% [markdown]
#
# ## Load Open-Ephys data


# %%

session = Session(os.path.join(datafolder, 
                               datatable['Npx-Folder'][0]))

def build_ttl_from_events(State, Sample):
    # we start at 0
    SN, TTL = [Sample[0]-30000], [0]
    # loop over events
    for state, sample in zip(State, Sample):
        if state==1:
            SN.append(sample); TTL.append(0)
            SN.append(sample); TTL.append(1)
        if state==0:
            SN.append(sample); TTL.append(1)
            SN.append(sample); TTL.append(0)
    # we force ending at 0
    SN.append(sample); TTL.append(0)
    SN.append(sample+30000); TTL.append(0)
    return np.array(SN, dtype=np.int32), np.array(TTL, dtype=np.uint8)

def load_OpenEphys(rec):

    # find TTL events on Probe A
    cond = (rec.events['stream_name']==PROBE_NAME)

    # load the events
    State = np.array(rec.events['state'][cond])
    Sample = np.array(rec.events['sample_number'][cond])
    pulse_onsets = Sample[State==1]

    # build the time array from the set of events
    SN, TTL = build_ttl_from_events(State, Sample)
    return pulse_onsets, SN, TTL 

print(' ==== PROTOCOLS FROM OPEN-EPHYS DATA ====  ')
props = []
iRec = 0
for r, rec in enumerate(session.recordnodes[NODE].recordings):

    pulse_onsets, SN, TTL = load_OpenEphys(rec)

    fig, ax = pt.figure(axes=(1,2), ax_scale=(2.5, 1.5), hspace=0)
    fig.suptitle('Recording #%i' % (r+1))
    ax[1].set_xlabel('N, sample number (Npx Probe)')
    ax[0].set_ylabel('TTL (all)'); ax[1].set_ylabel('splitted')
    pt.plot(SN, TTL, ax=ax[0])

    # tracking different protocols
    # --> more than 2s between protocols to identify protocol changes
    iStarts = np.concatenate([[0], 
                              np.flatnonzero(np.diff(SN)>(30e3*INTERPROTOCOL_WINDOW)),
                              [len(SN)]])

    for i0, i1 in zip(iStarts[:-1], iStarts[1:]):


        irange=np.arange(i0, np.min([i1+2,len(SN)]))
        pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])
       
        ax[1].plot(SN[irange], TTL[irange], lw=0.3, color=pt.tab10(iRec%10))
        pt.annotate(ax[1], 'protocol #%i'%(1+iRec) +iRec*'\n', (1,0), va='bottom', color=pt.tab10(iRec%10))

        DF.loc[iRec, 'i0'] = i0
        DF.loc[iRec, 'i1'] = i1
        DF.loc[iRec, 'Npx-Rec'] = 'node%i/exp%i/rec%i' % (NODE, EXP, r+1)
        DF.loc[iRec, 'ephys-nEpisodes'] = len(pulse_onsets[pulse_cond])

        iRec += 1

    pt.set_common_xlims(ax)
DF


# %%
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import least_squares


def find_sampling_match(t, nidaq_onsets, ephys_onsets):
    """
    we find the sample numbers that match the limits of the NIdaq acquisition,
    then, the samples in [nStart, nStop]
        have the time sampling:
            np.linspace(t[0], t[-1], nStop-nStart)

    where t is the nidaq time sampling array
    """

    N0 = ephys_onsets[0]
    t0 = nidaq_onsets[0]

    nMax = np.min([len(nidaq_onsets), len(ephys_onsets)])-1

    nMax=-1 # TO REMOVE 

    dN = ephys_onsets[nMax]-N0
    dT = nidaq_onsets[nMax]-t0

    if False:
        # IN CASE YOU WANT TO MAKE A MINIMIZATION FUNCTION, BUT FOR NOW not necessary...
        def to_minimize(x):
            T = (sn-x[0])*x[1]+t0
            func = interp1d(T, ttl) 
            probe_signal = func(np.clip(t, T.min(), T.max()))
            return np.mean((probe_signal-nidaqTTL)**2)

        res = least_squares(to_minimize, [N0, F0],
                            # max_nfev=10000, method='dogbox',
                            # ftol=None, xtol=None, verbose=True,
                            bounds=[(N0-3000, 0.99*F0), (N0+3000, 1.01*F0)])
        N0, F0 = res.x
    else:
        F0 = dT/dN

    nStart = N0-int(t0/F0)
    nStop = N0+dN+int((t[-1]-dT-t0)/F0) # we add dN to limit precision loss

    return nStart, nStop

def sampling_match(iRec,
                   with_fig=False):

    t, ephysSynch_signal, ephys_onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    

    # reload the open-ephys data:
    node = int(DF['Npx-Rec'][iRec].split('node')[1].split('/')[0])
    rec_id = int(DF['Npx-Rec'][iRec].split('rec')[1])-1
    rec = session.recordnodes[NODE].recordings[rec_id]
    # prepared ---> load
    pulse_onsets, SN, TTL = load_OpenEphys(rec)

    # restrict to previously identified range:
    irange=np.arange(DF['i0'][iRec], np.min([DF['i1'][iRec],len(SN)]))
    pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])

    # find the matching sample range
    nStart, nStop = find_sampling_match(t, ephys_onsets, pulse_onsets[pulse_cond])

    # we now match the time sampling in the data
    cond = (SN>=nStart) & (SN<=nStop)
    T = (SN[cond]-nStart)*(t[-1]-t[0])/(nStop-nStart)
    func = interp1d(T, TTL[cond], 
                    bounds_error=False,
                    fill_value=0)
    # we build a probe signal from the interpolation of the data
    probe_signal = func(t)

    width = 1.5
    if with_fig:

        fig, AX = pt.figure(axes=(4,2), ax_scale=(1.6,.7), top=1.5, hspace=1.6, wspace=0.3)
        fig.suptitle('protocol #%i (%i episodes)' % (iRec+1, np.sum(pulse_cond)))

        for i, t0 in enumerate([0.5, t[-1]/2+1, 3.*t[-1]/4., t[-1]]):

            pt.annotate(AX[0][i], 't=%.1fs' % t0, (0.1,1))

            # nidaq
            cond = (t>(t0-width)) & (t<(t0+width))
            AX[0][i].plot(t[cond][::10], ephysSynch_signal[cond][::10])
            pt.set_plot(AX[0][i], xlabel='NIdaq time (s)', ylabel='TTL\n(from NIdaq)' if i==0 else None)

            # open-ephys
            AX[1][i].plot(t[cond][::10], probe_signal[cond][::10])
            pt.set_plot(AX[1][i], xlabel='$F \\cdot (N- N_0) $ time (s)', ylabel='TTL\n(on Probe)' if i==0 else None)

            pt.set_common_xlims([AX[0][i], AX[1][i]])

        return nStart, nStop, fig
    else:
        return nStart, nStop

# sampling_match(1, with_fig=True)

# %%
#
for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _ =\
            sampling_match(iRec, with_fig=True)
DF

# %%

from physion.assembling.dataset import add_to_table

for key in ['Npx-Rec', 'nStart', 'nStop']:
    add_to_table(
        os.path.join(datafolder, 'DataTable0.xlsx'),
        sheet='Recordings',
        column=key,
        data=DF[key],
        insert_at=16 if 'nS' in key else 0)

# %%
#######################################################################################################



# %%
datafolder = os.path.expanduser('~/DATA/2026_02_13').replace('/', os.path.sep)

datatable

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
        session = Session(os.path.join(datafolder, datatable['Npx-Folder'][iRec]))

        node = int(DF['Npx-Rec'][iRec].split('node')[1].split('/')[0])
        rec_id = int(DF['Npx-Rec'][iRec].split('rec')[1])-1
        rec = session.recordnodes[NODE].recordings[rec_id]

        self.LFP = rec.continuous['ProbeA'].samples[nStart:nStop,:]

        # @Sally
        # self.spikes = ...


# %%
data = Data(datafolder, 3)

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

lfp_events = []
for e in events:
    cond = (data.t_probe>(e-1)) & (data.t_probe<(e+2))
    # lfp = gaussian_filter1d(data.LFP[cond,:].mean(axis=-1), 500)
    lfp = gaussian_filter1d(data.LFP[cond,0], 500)
    pre = (data.t_probe[cond]>(e-1)) & (data.t_probe[cond]<e)
    lfp_events.append(lfp-lfp[pre].mean())
t = data.t_probe[cond]-e

fig, ax = pt.figure(ax_scale=(2,3))
pt.plot(t, 1e-3*np.mean(lfp_events, axis=0), sy=1e-3*np.std(lfp_events, axis=0), ax=ax)
pt.set_plot(ax, xlabel='time from stim. (s)', ylabel='LFP (mV)')

# %%
datatable

# %%
