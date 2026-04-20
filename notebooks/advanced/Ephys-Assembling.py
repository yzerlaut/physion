
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
import sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
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

datafolder = os.path.expanduser('~/DATA/2026_04_10').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
PROBE_NAME = 'ProbeA'
EXP = 1 # 
NODE=0

# %% [markdown]
#
# ## Load Table data

# %%

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable0.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
DF = pd.DataFrame(columns=['time', 'Npx-Rec', 'daq-nEpisodes', 'ephys-nEpisodes', 'i0', 'i1', 'nStart', 'nStop'])
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
from scipy.interpolate import interp1d
import matplotlib.pylab as plt

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
        print('best shift found for, i=', i-Nshift)
    nidaq_onsets, ephys_onsets = nidaq_Onsets[:nMax], ephys_Onsets[i:nMax+i]

    N0 = ephys_onsets[0]
    t0 = nidaq_onsets[0]

    nMax = np.min([len(nidaq_onsets), len(ephys_onsets)])-2

    dN = ephys_onsets[-1]-N0
    dT = nidaq_onsets[-1]-t0
    F0 = dT/dN

    nStart = N0-int(t0/F0)
    nStop = N0+dN+int((t[-1]-dT-t0)/F0) # we add dN to limit precision loss

    return nStart, nStop


def sampling_match(iRec,
                   datafolder,
                   DF,
                   with_fig=False,
                   verbose=False):

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

    # find the matching sample range
    nStart, nStop = find_sampling_match(t, nidaq_onsets, pulse_onsets[pulse_cond],
                                        verbose=verbose)

    # we now match the time sampling in the data
    cond = (SN>=nStart) & (SN<=nStop)
    T = (SN[cond]-nStart)*(t[-1]-t[0])/(nStop-nStart)
    func = interp1d(T, TTL[cond], 
                    bounds_error=False,
                    fill_value=0)
    # we build a probe signal from the interpolation of the data
    probe_signal = func(t)

    width = 2.5
    if with_fig:
        import physion.utils.plot_tools as pt
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

        return nStart, nStop, fig, pulse_onsets[pulse_cond], nidaq_onsets
    else:
        return nStart, nStop
    

_, _, fig, ephys_onsets, nidaq_onsets = sampling_match(11, 
                                                       datafolder, DF,
                                                       with_fig=True, verbose=True)


# %%
#
for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _, _, _ =\
            sampling_match(iRec, 
                           session,
                           datafolder, DF,
                           with_fig=True)
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
