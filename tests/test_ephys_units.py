"""
single units: main channel of each unit
"""
import os, glob, types
import numpy as np
import pytest

from physion.analysis.modalities.ephys import EphysMixin


def test_main_channel_of_units():
    """ waveforms (time, channel, unit): unit u is large on channel main[u] """
    T, C, main = 61, 20, np.array([3, 17, 0, 9])
    rng = np.random.default_rng(0)
    W = 0.1*rng.standard_normal((T, C, len(main)))
    spike = -np.exp(-(np.arange(T)-20)**2/20.)
    for u, c in enumerate(main):
        W[:, c, u] += 5*spike
        W[:, (c+1)%C, u] += 2*spike        # neighbour channel, smaller
    data = types.SimpleNamespace(spikeWaveforms=W, has_spikeWaveforms=lambda: True)
    EphysMixin.find_main_channel_of_units(data)
    np.testing.assert_array_equal(data.main_channel_of_units, main)


NPX = (sorted(glob.glob(os.path.expanduser('~/DATA/Sally/Npx_WT_prelim_2026/NWBs/*.nwb')))+[''])[0]


@pytest.mark.skipif(not os.path.isfile(NPX), reason='no Neuropixels NWB file')
def test_main_channel_of_units_on_data():
    from physion.analysis.read_NWB import Data
    data = Data(NPX, verbose=False)
    data.find_main_channel_of_units()
    assert len(data.main_channel_of_units) == len(data.nwbfile.units)
    assert data.main_channel_of_units.max() < len(data.nwbfile.electrodes)
