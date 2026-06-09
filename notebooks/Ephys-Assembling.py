
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
# Then fill its neuropixels folder (`Npx-Folder`, for example 2026-03-19_16-13-00) and recordings information (`Npx-Rec`for example node0/exp1/rec1).    
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

datafolder = os.path.expanduser('~/DATA/2026_06_02').replace('/', os.path.sep)

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
iRec = 4 # example recording
_, _, fig, ephys_onsets, nidaq_onsets = sampling_match(iRec, 
                                                       datafolder, DF,
                                                       with_fig=True, verbose=True)


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
ELECTRODE_RANGE = [0,250]

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


# %%

# %%
