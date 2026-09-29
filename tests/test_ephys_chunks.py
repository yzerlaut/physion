"""
LFP and MUA computed in chunks (assembling/add_ephys.py)
    should match the spikeinterface pipeline on the whole recording
"""
import numpy as np
import pytest
from scipy import signal

si = pytest.importorskip('spikeinterface.full')

from physion.assembling.add_ephys import process_in_chunks, LFP_chunk, MUA_chunk

FS, FACTOR, NCH = 30000., 24, 8


def recording(duration=40.0003):
    """ noise + DC offset + slow oscillations (non-round number of samples) """
    rng = np.random.default_rng(0)
    n = int(duration*FS)
    t = np.arange(n)/FS
    x = rng.normal(0, 20, (n, NCH)) + 300 +\
            50*np.sin(2*np.pi*4*t)[:,None] + 30*np.sin(2*np.pi*80*t)[:,None]
    return si.NumpyRecording(x.astype(np.float32), sampling_frequency=FS)


def rel_diff(a, b, edge=2500):
    """ relative rms difference (excluding the edges) """
    k = min(len(a), len(b))
    a, b = a[edge:k-edge], b[edge:k-edge]
    return np.sqrt(np.mean((a-b)**2))/np.std(a)


def test_LFP_in_chunks():
    rec = recording()
    old = si.resample(si.bandpass_filter(rec, freq_min=0.5, freq_max=300.,
                                         ignore_low_freq_error=True, margin_ms=10000),
                      resample_rate=int(FS/FACTOR)).get_traces()
    new = process_in_chunks(rec, LFP_chunk([0.5, 300.], FACTOR), NCH,
                            resampling_factor=FACTOR, chunk_duration=3.)
    new = signal.sosfiltfilt(signal.butter(5, 0.5, btype='highpass', fs=FS/FACTOR,
                                           output='sos'), new, axis=0)
    # one sample per "timestamps[::resampling_factor]"
    assert len(new) == len(np.arange(rec.get_num_frames())[::FACTOR])
    assert rel_diff(old, new) < 0.01


def test_MUA_in_chunks():
    rec = recording()
    groups = [np.arange(0, 4), np.arange(4, 8)]
    hf = si.resample(si.rectify(si.bandpass_filter(rec, freq_min=300., freq_max=6000.)),
                     resample_rate=int(FS/FACTOR))
    old = np.array([hf.get_traces(channel_ids=hf.get_channel_ids()[g]).mean(axis=1)\
                        for g in groups]).T
    new = process_in_chunks(rec, MUA_chunk([300., 6000.], groups, FACTOR), len(groups),
                            resampling_factor=FACTOR, chunk_duration=3.)
    assert rel_diff(old, new) < 0.01
