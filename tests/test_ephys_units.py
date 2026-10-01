"""
single units: main channel and depth of each unit
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


class FakeTable(dict):
    """ minimal NWB table: columns accessed with ["column"][:] """
    @property
    def colnames(self): return list(self)


def fake_data(position_column, positions, rows):
    series = types.SimpleNamespace(electrodes=types.SimpleNamespace(data=np.array(rows)))
    nwbfile = types.SimpleNamespace(
        electrodes=FakeTable({position_column: np.array(positions, dtype=float)}),
        processing={'LFP': types.SimpleNamespace(data_interfaces={'LFP': series})})
    data = types.SimpleNamespace(nwbfile=nwbfile, df_name='test')
    data.depth_of_electrodes = lambda rows, key: \
            EphysMixin.depth_of_electrodes(data, rows, key)
    return data


@pytest.mark.parametrize('column', ['rel_y', 'y'])     # "y" in older files
def test_electrode_depths(column):
    """ 0 for the top electrode of the file, positive deeper """
    positions = np.arange(10)*20.            # probe channel 9 is the top one
    data = fake_data(column, positions, rows=[0, 4, 8])   # LFP: subset of channels
    np.testing.assert_array_equal(EphysMixin.electrode_depths(data, 'LFP'),
                                  [180., 100., 20.])


@pytest.mark.skipif(not os.path.isfile(NPX), reason='no Neuropixels NWB file')
def test_depth_of_LFP_and_MUA_on_data():
    from physion.analysis.read_NWB import Data
    data = Data(NPX, verbose=False)
    data.build_LFP()
    data.build_MUA()
    assert data.depth_LFP.shape == (data.LFP.shape[0],)
    assert data.depth_MUA.shape == (data.MUA.shape[0],)
    assert data.depth_LFP.min() >= 0


@pytest.mark.parametrize('column', ['rel_y', 'y'])
def test_depth_units(column):
    """ depth of the main channel of each unit, same reference as depth_LFP """
    data = fake_data(column, np.arange(10)*20., rows=[0, 4, 8])
    data.main_channel_of_units = np.array([9, 0, 4, 4])
    EphysMixin.build_depth_units(data)
    np.testing.assert_array_equal(data.depth_units, [0., 180., 100., 100.])


@pytest.mark.skipif(not os.path.isfile(NPX), reason='no Neuropixels NWB file')
def test_depth_units_on_data():
    from physion.analysis.read_NWB import Data
    data = Data(NPX, verbose=False)
    data.build_depth_units()
    data.build_LFP()
    assert data.depth_units.shape == (len(data.nwbfile.units),)
    assert data.depth_units.min() >= 0
    # same reference as the LFP channels
    assert data.depth_units.max() <= data.depth_of_electrodes(
                        np.arange(len(data.nwbfile.electrodes)), 'units').max()
