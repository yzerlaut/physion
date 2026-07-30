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
                            duration=0.3):
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

    IN PROGRESS:
    alternative function to the above, should be better

    """


    # rough initial guess for F:
    X0 = 32000. # 30kHz initial guess

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
