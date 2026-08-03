
# %% [markdown]
#
# # Assemble Neuropixels data
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```
# ## 1) 
# cd physion/src
# Run:
# ```
# python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA\2026_02_13
# ```
# this will create a file: `%USERPROFILE%\DATA\DataTable0.xlsx`  
#      move it to  ~/DATA/2026_02_13/DataTable0.xlsx
#
# Then fill its neuropixels folder (`Npx-Folder`, for example 2026-03-19_16-13-00) 
# and recordings information (`Npx-Rec`for example node0/exp1/rec1).    
#
#       N.B. you can use the code below to guide filling the recordings info

# %%
import sys, os
sys.path += [os.path.expanduser('~/physion/src'), '../src']
import numpy as np
import pandas as pd

from open_ephys.analysis import Session as OpenEphysSession
from physion.assembling.dataset import read_spreadsheet

################################################
###   code to do the alignement !!      ########
################################################
from physion.ephys.alignement import load_nidaq_synch_signal,\
    load_OpenEphys, sampling_match

import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_07_31').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
PROBE_NAME = 'ProbeA'
EXP = 1 # 
NODE=0

# %% [markdown]
#
# ## Load Table data

# %%

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
DF = pd.DataFrame(columns=['time', 'Npx-Rec',\
                           'daq-nEpisodes', 'ephys-nEpisodes', 
                           'i0', 'i1', 'nStart', 'nStop',
                           'LFP', 'MUA', 'Spikes', 'raw-Ephys',
                           'electrode-range'])
DF['time'] = datatable['time']
DF['Npx-Folder'] = datatable['Npx-Folder']

# loop over protocols
# print(' ==== PROTOCOLS FROM NIDAQ DATA ====  ')
for iRec, protocol in enumerate(datatable['protocol']):
    _, _, onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    print(' rec #%i) n=%i episodes, %s' % (iRec+1, len(onsets), protocol))
    DF.loc[iRec, 'daq-nEpisodes'] = len(onsets)

# %% [markdown]
#
# ## Load Open-Ephys data

# %%
sessions = []
for folder in datatable['Npx-Folder'].unique():
    sessions.append(\
        OpenEphysSession(os.path.join(datafolder, folder)))


# %%

print(' ==== PROTOCOLS FROM OPEN-EPHYS DATA ====  ')
props = []
iRec = 0
for session in sessions:
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
            print(iRec, len(irange))
            if len(irange)>10:
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

import os
import numpy as np
import json
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from physion.acquisition.tools import find_line_props
from open_ephys.analysis import Session as OpenEphysSession

def load_nidaq_synch_signal(folder):
    """ 
    """

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

def build_trace_from_events(events, 
                            t_array,
                            duration=0.1):
    """
    converts a set of onset-events into a time trace
        (set the duration of events thourgh "duration")
    """
    output = np.zeros(len(t_array), dtype=bool)
    # loop over events
    for e in events:
        cond = (t_array>e) & (t_array<(e+duration))
        output[cond] = True
    return output

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

def load_OpenEphys(rec,
                   PROBE_NAME='ProbeA'):

    # find TTL events on Probe A
    cond = (rec.events['stream_name']==PROBE_NAME) &\
                (rec.events['sample_number']>0)

    # load the events
    State = np.array(rec.events['state'][cond])
    Sample = np.array(rec.events['sample_number'][cond])
    pulse_onsets = Sample[State==1]

    # build the time array from the set of events
    SN, TTL = build_ttl_from_events(State, Sample)

    return pulse_onsets, SN, TTL 

def find_sampling_match(t, nidaq_Onsets, ephys_Onsets,
                        Nshift=20, verbose=False):
    """
    we find the sample numbers that match the limits of the NIdaq acquisition,
    then, the samples in [nStart, nStop]
        have the time sampling:
            np.linspace(t[0], t[-1], nStop-nStart)

    where t is the nidaq time sampling array

    Because some TTL events can appear without being triggered by the NIdaq
    we test different shifts to find the right alignement and we take the best !
        --> to be checked visually in the figure !
    """

    # varying the shift and computing correlations
    CC, nMax = [], len(nidaq_Onsets)-int(2*Nshift)
    nMax = np.min([len(nidaq_Onsets), len(ephys_Onsets)])-int(2*Nshift)

    # print(len(nidaq_Onsets), len(ephys_Onsets))
    for i in range(2*Nshift):
        CC.append(np.corrcoef(nidaq_Onsets[:nMax], ephys_Onsets[i:nMax+i])[0,1])

    # finding the best correlation between times:
    i = int(np.argmax(CC))
    if verbose:
        print('best shift found for, i=', i)
    nidaq_onsets, ephys_onsets = nidaq_Onsets[:nMax], ephys_Onsets[i:nMax+i]

    N0 = ephys_onsets[0]
    t0 = nidaq_onsets[i]

    nMax = np.min([len(nidaq_onsets), len(ephys_onsets)])-2

    dN = ephys_onsets[-1]-N0
    dT = nidaq_onsets[-1]-t0
    F0 = dT/dN

    nStart = N0-int(t0/F0)
    nStop = N0+dN+int((t[-1]-dT-t0)/F0) # we add dN to limit precision loss

    return nStart, nStop


def find_match(t, nidaq_Onsets, ephys_Onsets,
               index_first_event=0, # vary if problems, you might be unlucky and this step is missing on the probe
               Nshifts=10,
               with_residual_fig=True,
               verbose=True):
    """

    Nshifts should be larger than Nsecurity
    """


    # rough initial guess for F:
    X0 = 30000. # 30kHz initial guess

    # allowing shifts of onset times with secure bounds:
    nidaq_Onsets_secure = nidaq_Onsets[Nshifts:-Nshifts]
    n = len(nidaq_Onsets_secure)
    
    t0 = nidaq_Onsets_secure[index_first_event] # always fixed, time of first event

    residuals, coeffs, shifts, N0s = [], [], [], []

    rdm_pick = np.random.choice(np.arange(n), 30)
    def to_minimize(X, shift):
        iStart = Nshifts+shift
        N0 = ephys_Onsets[iStart] # sample of first event
        return np.abs(\
            (ephys_Onsets[iStart:iStart+n]-N0)/X[0]\
                -(nidaq_Onsets_secure-t0)).mean()

    for i in range(-Nshifts, Nshifts+1):
        res = minimize(to_minimize, X0, args=(i), tol=1e-8)
        residuals.append(res.fun)
        coeffs.append(res.x)
        shifts.append(i)
        N0s.append(ephys_Onsets[Nshifts+i])

    if with_residual_fig:
        fig, ax = pt.figure(ax_scale=(2,1))
        ax.plot(shifts, residuals, 'wo')
        ax.set_ylim([0,1])
        pt.set_plot(ax, xlabel='+step shift', ylabel='residual')

    iMin = np.argmin(residuals)
    res = minimize(to_minimize, X0, args=shifts[iMin], tol=1e-8)
    print(res)
    N0, F = N0s[iMin], 1./res.x[0]

    # dN = ephys_onsets[-1]-N0
    # dT = nidaq_onsets[-1]-t0
    # F0 = dT/dN

    nStart = N0-int(t0/F)
    nStop = nStart+int((t[-1]-t[0])/F) # we add dN to limit precision loss
    return nStart, nStop

def sampling_match(iRec,
                   datafolder,
                   DF,
                   with_fig=True,
                   verbose=False,
                   width=2.5):

    session = OpenEphysSession(os.path.join(datafolder, DF['Npx-Folder'][iRec]))

    t, ephysSynch_signal, nidaq_onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, DF['time'][iRec]))

    # reload the open-ephys data:
    node = int(DF['Npx-Rec'][iRec].split('node')[1].split('/')[0])
    rec_id = int(DF['Npx-Rec'][iRec].split('rec')[1])-1
    rec = session.recordnodes[node].recordings[rec_id]
    # prepared ---> load
    pulse_onsets, SN, TTL = load_OpenEphys(rec)

    # restrict to previously identified range:
    irange=np.arange(DF['i0'][iRec], np.min([DF['i1'][iRec],len(SN)]))
    pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])

    nStart, nStop = find_sampling_match(t, nidaq_onsets, pulse_onsets[pulse_cond],
                                        verbose=verbose)
    # nStart, nStop = find_match(t, nidaq_onsets, pulse_onsets[pulse_cond],
    #                                     verbose=verbose)
    F = (t[-1]-t[0])/(nStop-nStart)

    if with_fig:

        def nidaq_to_probe(t):
            return nStart+t/F
        
        import physion.utils.plot_tools as pt
        fig, AX = pt.figure(axes=(4,2), ax_scale=(1.6,.7), top=1.5, hspace=1.6, wspace=0.3)
        fig.suptitle('protocol #%i (%i episodes)' % (iRec+1, np.sum(pulse_cond)))

        for i, t0 in enumerate([width, t[-1]/2+1, 3.*t[-1]/4., t[-1]-width]):

            pt.annotate(AX[0][i], 't=%.1fs' % (t0-width), (0.1,1))

            # nidaq
            cond = (t>(t0-width)) & (t<(t0+width))
            AX[0][i].plot(t[cond][::10], ephysSynch_signal[cond][::10])
            pt.set_plot(AX[0][i],
                        xlabel='NIdaq time (s)', 
                        xlim=[t0-width,t0+width],
                        ylabel='TTL\n(from NIdaq)' if i==0 else None)

            # open-ephys
            # AX[1][i].plot(t[cond][::10], probe_signal[cond][::10])
            # pt.set_plot(AX[1][i], xlabel='$F \\cdot (N- N_0) $ time (s)', ylabel='TTL\n(on Probe)' if i==0 else None)
            AX[1][i].plot(SN, TTL)
            pt.set_plot(AX[1][i], 
                        xlim=nidaq_to_probe(np.array([t0-width,t0+width])),
                        xlabel='probe sample', ylabel='TTL\n(on Probe)' if i==0 else None)

            #pt.set_common_xlims([AX[0][i], AX[1][i]])

        return nStart, nStop, fig, pulse_onsets[pulse_cond], nidaq_onsets
    else:
        return nStart, nStop



iRec = 3 # example recording
_, _, fig, ephys_onsets, nidaq_onsets = sampling_match(iRec, 
                                                       datafolder, DF,
                                                       with_fig=True,
                                                       verbose=True)

# %%

F0 = 30000.

Residuals, N0s, t0s, Fs = [], [], [], []

iStepNIDAQ= 0
iStepEphys = 0
t0 = nidaq_onsets[iStepNIDAQ]
new_nidaq_onsets = nidaq_onsets-t0

dt=1e-2
t = np.arange(int(new_nidaq_onsets[-1]/dt))*dt
t = t[t>0.8*t.max()]
nidaq_trace = build_trace_from_events(new_nidaq_onsets, t)

def to_minimize(F):
    N0 = ephys_onsets[iStepEphys]
    new_ephys_onsets = (ephys_onsets-N0)/F
    ephys_trace = build_trace_from_events(new_ephys_onsets, t)
    return 1-np.corrcoef(nidaq_trace, ephys_trace)[0,1]

res = minimize(to_minimize, [F0])
print(res)
print(res.x)
N0 = ephys_onsets[iStepEphys]
new_ephys_onsets = (ephys_onsets-N0)/res.x
ephys_trace = build_trace_from_events(new_ephys_onsets, t)

fig, ax = pt.figure(ax_scale=(2,1))
cond = (t>(t[-1]-5))
ax.plot(t[cond], nidaq_trace[cond])
ax.plot(t[cond], 1+ephys_trace[cond])

#%%
    # if verbose:
    #     print('best shift correspond to, N0=%i F=%.4e' % (N0, F))
    # return t0, N0, F, shifts[iMin], res

# t0, N0, F, shift, res = find_match(nidaq_onsets, pulse_onsets[pulse_cond])
iRec = 2 # example recording
_, _, fig, ephys_onsets, nidaq_onsets = find_sampling_match(iRec, 
                                                       datafolder, DF,
                                                    #    with_fig=True,
                                                    #    verbose=True)


# %%
#print(nidaq_onsets[0])
# F*(pulse_onsets[pulse_cond][shift]-N0)


# %%
#
for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _, _, _ =\
            sampling_match(iRec, 
                           datafolder, DF,
                           with_fig=True)
DF

# %%

from physion.assembling.dataset import add_to_table

for key in ['Npx-Rec', 'nStart', 'nStop']+\
           ['LFP', 'MUA', 'Spikes', 'raw-Ephys']:
    add_to_table(
        os.path.join(datafolder, 'DataTable.xlsx'),
        sheet='Recordings',
        column=key,
        insert_at=16,
        data=DF[key])
DF
# %%
# TODO add a column with sub-selection of electrode range !!
ELECTRODE_RANGE = [0,200]

# %%
add_to_table(
    os.path.join(datafolder, 'DataTable.xlsx'),
    sheet='Recordings', column='electrode-range',
    insert_at=20,
    data=['%i-%i' % (ELECTRODE_RANGE[0], ELECTRODE_RANGE[1]) for _ in range(len(DF['Npx-Rec']))])
add_to_table(
    os.path.join(datafolder, 'DataTable.xlsx'),
    sheet='Recordings', column='electrode-subsampling',
    insert_at=21,
    data=[40 for _ in range(len(DF['Npx-Rec']))])

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
    pt.set_plot(ax, xlabel='time (s)', ylabel='uV',
                ylim=[-200,200])
    return fig, ax

iRecs = np.flatnonzero(datatable['protocol']=='flashed-stimuli')

for iRec in iRecs:

    t, LFP = load_LFP_resp(datafolder, iRec, channel_subsampling=4,
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
