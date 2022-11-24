import numpy as np
import os, time
from scipy.signal import argrelextrema
from scipy.ndimage.filters import gaussian_filter1d

def realign_from_photodiode(signal,
                            metadata,
                            sampling_rate=None,
                            photodiode_rise_time=0.01,
                            shift_time=0.3, # MODIFY IT HERE IN CASE NEEDED
                            debug=False, istart_debug=0, n_vis=5,
                            indices_forced=[],
                            times_forced=[],
                            durations_forced=[],
                            verbose=True):
    """
    

    - shift_time is to handle the fact that there might be a delay in stopping the protocol 
    without this, the start of the next one is put as the end of the previous...
    """
    if verbose:
        print('---> Realigning data with respect to photodiode signal [...] ')

    success = True

    # extract parameters
    if sampling_rate is None:
        dt = 1./metadata['NIdaq-acquisition-frequency']
    else:
        dt = 1/sampling_rate
    t = np.arange(len(signal))*dt
    
    tlim, tnew = [0, t[-1]], 0

    tstart, tshift = metadata['time_start'][0]-1, 0
    metadata['time_start_realigned'] = []

    if verbose:
        print('smoothing photodiode signal [...]')

    # smoothing the signal
    smooth_signal = np.diff(gaussian_filter1d(np.cumsum(signal), 20)) # integral + smooth + derivative
    smooth_signal[:1000], smooth_signal[-10:] = smooth_signal[1000], smooth_signal[-1000] # to insure no problem at borders (of the derivative)

    # compute signal boundaries to evaluate threshold crossing of photodiode signal
    H, bins = np.histogram(smooth_signal, bins=100)
    baseline = bins[np.argmax(H)+1]
    threshold = (np.max(smooth_signal)-baseline)/4. # reaching 25% of peak level

    # looping over episodes
    i=0
    while (i<len(metadata['time_duration'])) and (tstart<(t[-1]-metadata['time_duration'][i])) and success:

        # the next time point above being above threshold
        cond_thresh = (t[:-2]>tstart+shift_time) & (smooth_signal[1:]>=(baseline+threshold)) & (smooth_signal[:-1]<(baseline+threshold))
        # print(tstart, i, success)

        if i in indices_forced:
            iforced = np.argwhere(np.array(indices_forced)==i)[0][0]
            # possibility to force some indices not to be re-aligned
            print('forced index %i to start at time %.1f for a duration %.1f' % (i, times_forced[iforced], durations_forced[iforced]))
            metadata['time_duration'][i] = durations_forced[iforced]
            metadata['time_start_realigned'].append(times_forced[iforced])
            tstart = times_forced[iforced]
            tshift = 0.5

        elif np.sum(cond_thresh)>0:
            # success
            tshift = t[:-2][cond_thresh][0] - tstart - photodiode_rise_time
            metadata['time_start_realigned'].append(tstart+tshift)
        else:
            success = False
            # we don't do anything, we just increment the episode id
            print('realignment stopped from episode #%i !' % i)
        

        if success and debug and ((i>=istart_debug) and (i<istart_debug+n_vis)):
            # plot alignment for debugging !
            cond = (t[:-1]>=tstart+shift_time-5) & (t[:-1]<=tstart+tshift+10)
            fig, ax = plt.subplots()
            ax.plot(t[:-1][cond], signal[:-1][cond], label='signal')
            ax.plot(t[:-1][cond], smooth_signal[cond], label='smoothed')
            ax.plot((tstart+tshift)*np.ones(2), ax.get_ylim(), 'k:', label='onset')
            ax.plot(ax.get_xlim(), (baseline+threshold)*np.ones(2), 'k:', label='threshold')
            ax.plot(ax.get_xlim(), baseline*np.ones(2), 'k:', label='baseline')
            ax.plot((tstart+tshift+metadata['time_duration'][i])*np.ones(2), ax.get_ylim(), 'k:', label='offset')
            plt.xlabel('time (s)')
            plt.ylabel('norm. signals')
            ax.set_title('ep. #%i' % i)
            ax.legend(frameon=False)
            plt.show()

        if success:
            tstart=tstart+tshift+metadata['time_duration'][i] # update tstart by tshift_observed+duration
            i+=1
            

    # transform to numpy array
    metadata['time_start_realigned'] = np.array(metadata['time_start_realigned'])
    metadata['time_stop_realigned'] = metadata['time_start_realigned']+\
        metadata['time_duration'][:len(metadata['time_start_realigned'])]
    
    # if the protocol is not complete, the last one might be truncated, we remove it !
    if len(metadata['time_start_realigned'])<len(metadata['time_start']):
        metadata['time_start_realigned'] = metadata['time_start_realigned'][:-1]
        metadata['time_stop_realigned'] = metadata['time_stop_realigned'][:-1]
        
    if verbose:
        print('[ok]          --> succesfully realigned')
        print('                  found n=%i episodes over the %i of the protocol ' % (len(metadata['time_start_realigned']), len(metadata['time_start'])))
            
    return True, metadata


def find_onset_time(t, photodiode_signal,
                    baseline=0, high_level=1):
    """
    """
    cond = (photodiode_signal[1:]>=(0.5*high_level)) & (photodiode_signal[:-1]<=(0.5*high_level))
    if np.sum(cond)>0:
        return t[:-1][cond][0]
    else:
        return None
    

# def find_onset_time(t, photodiode_signal,
#                     smoothing_time = 20e-3,
#                     # advance_time = 15e-3,
#                     baseline=0, high_level=1):
#     """
#     we smooth the photodiode signal, with a gaussian filter of extent Tsmoothing
#     Tonset = Tcrossing-3./4.*Tsmoothing
#     Tcrossing is the time of crossing of half the max-min level (of the smoothed signal)
#     """
#     advance_time = 3./4.*smoothing_time
#     smoothed = gaussian_filter1d(photodiode_signal, int(smoothing_time/(t[1]-t[0])))
#     smoothed = (smoothed-smoothed.min())/(smoothed.max()-smoothed.min())
#     cond = (smoothed[1:]>=0.5) & (smoothed[:-1]<=0.5)
#     t0 = t[:-1][cond][0]
#     return t0-advance_time, smoothed, smoothed.min()+0.5*(smoothed.max()-smoothed.min())


def normalize_signal(x):
    # just to plot above
    norm, xmin = 1./(np.max(x)-np.min(x)), np.min(x)
    return norm*(x-xmin), norm, xmin

if __name__=='__main__':

    import matplotlib.pylab as plt

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Realigning visual stimulation episodes from Photodiode signals
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datafolder")
    parser.add_argument('-n', "--n_vis", type=int, default=5)
    parser.add_argument('-id', "--istart_debug", type=int, default=0)
    parser.add_argument("--smoothing_time", type=float, help='in s', default=20e-3)
    parser.add_argument('-st', "--shift_time", type=float, help='in s', default=0.3)
    parser.add_argument("--indices_forced", nargs='*', type=int, default=[])
    parser.add_argument("--times_forced", nargs='*', type=float, default=[])
    parser.add_argument("--durations_forced", nargs='*', type=float, default=[])
    args = parser.parse_args()

    data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()['analog'][0]
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'), allow_pickle=True).item()
    VisualStim = np.load(os.path.join(args.datafolder, 'visual-stim.npy'), allow_pickle=True).item()

    if 'time_duration' not in VisualStim:
        VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])
    for key in VisualStim:
        metadata[key] = VisualStim[key]

    # plt.plot(data[::1000][:1000])
    # plt.title('photodiode-signal (subsampled/100)')
    # plt.show()

    realign_from_photodiode(data, metadata,
                            debug=True,
                            istart_debug=args.istart_debug,
                            shift_time=args.shift_time,
                            n_vis=args.n_vis,
                            indices_not_realigned=args.indices_not_realigned,
                            verbose=True)
    









