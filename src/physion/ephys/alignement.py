import os
import numpy as np
import json
from scipy.interpolate import interp1d

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
