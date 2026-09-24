"""
realignement of the visual stimulation episodes on the photodiode signal
    and storage of the episode parameters in the NWB file
"""
import types, datetime
import numpy as np
import pynwb
import pytest

from physion.assembling.realign_from_photodiode import realign_from_photodiode
from physion.assembling.nwb import add_visual_stimulation

RATE = 1000.  # Hz
DELAY = 0.08  # s, screen delay of the photodiode onsets
DURATIONS = np.array([2., 2.5, 1.5, 2., 3., 2., 1.8, 2.2])
ANGLES = np.arange(len(DURATIONS))*22.5   # one value per episode


def protocol_and_photodiode():
    """ episodes separated by 4s of blank screen (the baseline is the most frequent level) """
    time_start = 2. + np.concatenate([[0], np.cumsum(DURATIONS+4)[:-1]])
    time_stop = time_start + DURATIONS
    t = np.arange(0, time_stop[-1]+3, 1./RATE)
    signal = np.zeros(len(t))
    for t0, t1 in zip(time_start, time_stop):
        signal[(t>=t0+DELAY) & (t<t1+DELAY)] = 1.
    return time_start, time_stop, signal


def test_realign_with_excluded_episode():
    time_start, time_stop, signal = protocol_and_photodiode()
    metadata = {'time_start':time_start, 'time_duration':DURATIONS.copy()}
    success, metadata = realign_from_photodiode(signal, metadata, sampling_rate=RATE,
                                                exclude_episodes=[3], verbose=False)
    kept = np.array([0, 1, 2, 4, 5, 6, 7])  # the last episode is not dropped
    np.testing.assert_array_equal(metadata['realigned_episode_indices'], kept)
    np.testing.assert_allclose(metadata['time_start_realigned'],
                               time_start[kept]+DELAY, atol=0.02)
    np.testing.assert_allclose(metadata['time_stop_realigned']-metadata['time_start_realigned'],
                               DURATIONS[kept])


def build(tmp_path, exclude_episodes=[], metadata_extra={}):
    """ add_visual_stimulation on a synthetic session """
    time_start, time_stop, signal = protocol_and_photodiode()
    np.save(tmp_path/'visual-stim.npy',
            {'time_start':time_start, 'time_stop':time_stop,
             'index':np.arange(len(DURATIONS)), 'protocol_id':np.zeros(len(DURATIONS), dtype=int),
             'angle':ANGLES})
    metadata = {'VisualStim':True, 'protocol':'test',
                'NIdaq':{'acquisition-frequency':RATE,
                         'analog-inputs':{'channel-labels':['photodiode-signal-from-screen']}}}
    metadata.update(metadata_extra)
    args = types.SimpleNamespace(datafolder=str(tmp_path), modalities=['VisualStim'],
                    photodiode_sampling=RATE, reverse_photodiodeSignal=False,
                    force_to_visualStimTimestamps=False, max_episode=-1,
                    exclude_episodes=exclude_episodes, indices_forced=None,
                    times_forced=None, durations_forced=None, verbose=False)
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
            session_start_time=datetime.datetime.now(datetime.timezone.utc))
    add_visual_stimulation(nwbfile, metadata, {'analog':[signal]}, args)
    return {k: np.array(nwbfile.stimulus[k].data).flatten()
                for k in ['time_start_realigned', 'time_stop_realigned', 'angle', 'index']}


def test_episode_parameters_without_exclusion(tmp_path):
    stim = build(tmp_path)
    np.testing.assert_array_equal(stim['angle'], ANGLES)
    np.testing.assert_allclose(stim['time_stop_realigned']-stim['time_start_realigned'], DURATIONS)


def test_episode_parameters_match_times_with_excluded_episode(tmp_path):
    stim = build(tmp_path, exclude_episodes=[3])
    kept = np.array([0, 1, 2, 4, 5, 6, 7])
    np.testing.assert_array_equal(stim['index'], kept)
    np.testing.assert_array_equal(stim['angle'], ANGLES[kept])
    np.testing.assert_allclose(stim['time_stop_realigned']-stim['time_start_realigned'],
                               DURATIONS[kept])


def test_forced_realignement_from_metadata(tmp_path):
    """ "realignement_..._forced" of metadata.json: episode 0 forced at t=1.5s """
    stim = build(tmp_path, metadata_extra={'realignement_indices_forced':[0],
                                           'realignement_times_forced':[1.5],
                                           'realignement_durations_forced':[2.]})
    assert stim['time_start_realigned'][0] == 1.5
    assert len(stim['index']) == len(DURATIONS)
