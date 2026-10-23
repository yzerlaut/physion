"""
Quality check of the visual stimulation presentation of a recording

    python -m physion.visual_stim.check_presentation path/to/datafolder [--plot]

Uses:
    - "visual-stim-frames.npy" : log of the presentation process (frame displayed at each refresh)
    - "movie-frames.npy"       : frame table of the movie (state of the monitoring square in each frame)
    - "NIdaq.npy" + "metadata.npy" (optional): the recorded photodiode signal

With a single photodiode, the log and the frame table predict the photodiode
signal at each refresh: the recorded signal is compared to this prediction to
measure the display latency and to find the square transitions that did not
happen when the log says they did (e.g. frames dropped by the graphics driver
after the buffer swap, which the presentation process cannot see).
"""
import os, sys
import numpy as np

from physion.visual_stim.presenter import LOG_FILENAME, FLAG_UNDERRUN, FLAG_ADVANCE,\
        FLAG_DELAY, FLAG_FORCED, FLAG_MISSED_VSYNC
from physion.visual_stim.movie import load_frame_table


def load_log(datafolder):
    fn = os.path.join(datafolder, LOG_FILENAME)
    if not os.path.isfile(fn):
        raise FileNotFoundError('no presentation log ("%s") in "%s"' % (LOG_FILENAME, datafolder))
    return np.load(fn, allow_pickle=True).item()


def expected_square(log, table):
    """ expected square state at each flip (-1 when unknown) """
    frames = np.asarray(log['frame'], dtype=int)
    square = np.full(len(frames), -1, dtype=int)
    if table is not None:
        ok = frames>=0
        square[ok] = table['square'][frames[ok] % int(table['n_frames'])]
    return square


def load_photodiode(datafolder):
    """ returns (t, signal) from the NIdaq recording, or None """
    fn = os.path.join(datafolder, 'NIdaq.npy')
    if not os.path.isfile(fn):
        return None
    data = np.load(fn, allow_pickle=True).item()
    chan = 0
    mfn = os.path.join(datafolder, 'metadata.npy')
    if os.path.isfile(mfn):
        metadata = np.load(mfn, allow_pickle=True).item()
        try:
            labels = np.array(metadata['NIdaq']['analog-inputs']['channel-labels'])
            chan = int(np.flatnonzero(labels=='photodiode-signal-from-screen')[0])
        except (KeyError, IndexError, TypeError):
            pass
    signal = np.asarray(data['analog'][chan], dtype=float)
    t = np.arange(len(signal))*float(data['dt'])
    return t, signal


def square_transitions(t_flip, square):
    """ times of the expected square changes (0->1: +1, 1->0: -1) """
    valid = square>=0
    t, s = t_flip[valid], square[valid]
    i = np.flatnonzero(np.diff(s)!=0)+1
    return t[i], np.sign(s[i]-s[i-1])


def detect_transitions(t, signal, smoothing=0.002):
    """ threshold crossings of the photodiode signal (mid-level) """
    from scipy.ndimage import gaussian_filter1d
    dt = t[1]-t[0]
    x = gaussian_filter1d(signal, max([1., smoothing/dt]))
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    if hi-lo<=0:
        return np.array([]), np.array([]), x
    thr = (lo+hi)/2.
    above = x>thr
    i = np.flatnonzero(np.diff(above.astype(int))!=0)+1
    return t[i], np.where(above[i], 1, -1), x


def compare(expected_t, expected_sign, recorded_t, recorded_sign,
            max_latency=0.2):
    """
    display latency (median delay between expected and recorded transitions)
        and the expected transitions without a matching recorded one

    returns latency, delays (one per expected transition, nan if no match)
    """
    delays = np.full(len(expected_t), np.nan)
    for sign in [-1, 1]:
        rec = recorded_t[recorded_sign==sign]
        if len(rec)==0:
            continue
        for i in np.flatnonzero(expected_sign==sign):
            j = np.searchsorted(rec, expected_t[i]-0.01)
            if (j<len(rec)) and (rec[j]-expected_t[i]<max_latency):
                delays[i] = rec[j]-expected_t[i]
    latency = float(np.nanmedian(delays)) if np.any(np.isfinite(delays)) else np.nan
    return latency, delays


def check(datafolder, plot=False, verbose=True):

    log = load_log(datafolder)
    table = load_frame_table(datafolder)
    t_flip = np.asarray(log['flip_time'])
    flags = np.asarray(log['flags'])
    T = log.get('refresh_period_estimate', 1./60)

    report = {'summary':log.get('summary', {}),
              'vsync':log.get('vsync'),
              'refresh_rate':1./T}

    events = {'missed vsync':t_flip[(flags & FLAG_MISSED_VSYNC)>0],
              'decoder underrun':t_flip[(flags & FLAG_UNDERRUN)>0],
              'forced correction':t_flip[(flags & FLAG_FORCED)>0],
              'drift correction':t_flip[((flags & (FLAG_ADVANCE|FLAG_DELAY))>0) & ((flags & FLAG_FORCED)==0)]}
    report['events'] = events

    intervals = np.diff(t_flip)
    report['flip_interval_ms'] = {'median':1e3*float(np.median(intervals)) if len(intervals) else np.nan,
                                  'max':1e3*float(np.max(intervals)) if len(intervals) else np.nan}

    square = expected_square(log, table)
    exp_t, exp_s = square_transitions(t_flip, square)
    report['n_expected_square_transitions'] = len(exp_t)

    PD = load_photodiode(datafolder)
    if (PD is not None) and len(exp_t)>0:
        t, signal = PD
        rec_t, rec_s, smoothed = detect_transitions(t, signal)
        # the photodiode can be inverted (depends on the rig)
        lat1, d1 = compare(exp_t, exp_s, rec_t, rec_s)
        lat2, d2 = compare(exp_t, exp_s, rec_t, -rec_s)
        if np.sum(np.isfinite(d2))>np.sum(np.isfinite(d1)):
            latency, delays, rec_s = lat2, d2, -rec_s
        else:
            latency, delays = lat1, d1
        report['display_latency_ms'] = 1e3*latency
        late = np.isfinite(delays) & (np.abs(delays-latency)>0.75*T)
        report['n_missing_transitions'] = int(np.sum(~np.isfinite(delays)))
        report['n_mistimed_transitions'] = int(np.sum(late))
        report['missing_transitions_t'] = exp_t[~np.isfinite(delays)]
        report['mistimed_transitions_t'] = exp_t[late]
        # (robust estimate: median absolute deviation)
        report['transition_jitter_ms'] = 1e3*1.4826*float(np.nanmedian(np.abs(delays-latency)))
        report['photodiode_matches'] = bool(np.mean(np.isfinite(delays) & ~late)>0.5)
    else:
        PD = None

    if verbose:
        print_report(report)

    if plot:
        plot_report(report, log, t_flip, square, PD, datafolder)

    return report


def print_report(report):
    S = report['summary']
    print('\n ---- visual stimulation presentation ----')
    print('  screen: %.2f Hz, vsync: %s' % (report['refresh_rate'], report['vsync']))
    print('  flip intervals: median %.2f ms, max %.1f ms' % (\
            report['flip_interval_ms']['median'], report['flip_interval_ms']['max']))
    for key in ['n_frames_shown', 'n_skipped_frames', 'n_corrections', 'max_abs_drift_ms']:
        if key in S:
            print('  %s: %s' % (key, S[key]))
    for key, times in report['events'].items():
        if len(times)>0:
            print('  %s: n=%i, at t=%s%s' % (key, len(times),
                    ', '.join('%.2fs' % x for x in times[:8]), ' ...' if len(times)>8 else ''))
        else:
            print('  %s: none' % key)
    if 'display_latency_ms' in report:
        print('  --- photodiode ---')
        print('  display latency: %.1f ms (jitter: %.1f ms)' % (\
                report['display_latency_ms'], report['transition_jitter_ms']))
        print('  expected square transitions: %i' % report['n_expected_square_transitions'])
        print('     - missing in the photodiode signal: %i %s' % (report['n_missing_transitions'],
                ', '.join('%.2fs' % x for x in report['missing_transitions_t'][:8])))
        print('     - mistimed (>0.75 refresh): %i %s' % (report['n_mistimed_transitions'],
                ', '.join('%.2fs' % x for x in report['mistimed_transitions_t'][:8])))
        if not report['photodiode_matches']:
            print('  [!!] most transitions do not match: wrong photodiode channel, no monitoring')
            print('       square on this screen, or a demo recording ? [!!]')
    else:
        print('  (no photodiode signal compared)')
    print('')


def plot_report(report, log, t_flip, square, PD, datafolder):
    import matplotlib.pylab as plt
    fig, AX = plt.subplots(3 if PD is not None else 2, 1, figsize=(10, 6), sharex=True)
    AX[0].plot(t_flip[1:], 1e3*np.diff(t_flip), 'k-', lw=0.5)
    AX[0].set_ylabel('flip interval (ms)')
    AX[1].plot(t_flip, log['drift'], 'k-', lw=0.5)
    AX[1].set_ylabel('drift (ms)')
    colors = {'missed vsync':'tab:red', 'decoder underrun':'tab:orange',
              'forced correction':'tab:purple', 'drift correction':'tab:green'}
    for key, times in report['events'].items():
        for ax in AX:
            for x in times[:200]:
                ax.axvline(x, color=colors[key], lw=0.5, alpha=0.5)
    if PD is not None:
        t, signal = PD
        lat = report.get('display_latency_ms', 0)/1e3
        AX[2].plot(t, (signal-np.min(signal))/(np.ptp(signal)+1e-12), 'k-', lw=0.5,
                   label='photodiode')
        AX[2].step(t_flip+lat, np.clip(square, 0, 1), 'c-', where='post', lw=0.8,
                   label='expected (+latency)')
        for x in report['missing_transitions_t'][:200]:
            AX[2].axvline(x, color='tab:red', lw=1)
        AX[2].legend(loc='upper right', frameon=False)
    AX[-1].set_xlabel('time from acquisition start (s)')
    fig.tight_layout()
    fn = os.path.join(datafolder, 'visual-stim-presentation.png')
    fig.savefig(fn, dpi=150)
    print('  -> figure saved as: %s' % fn)
    plt.show()


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("datafolder")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    check(args.datafolder, plot=args.plot)
