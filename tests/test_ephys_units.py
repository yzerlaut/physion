"""
single units: main channel and depth of each unit,
    restriction of the ephys data to a brain region
"""
import os, glob, types, datetime
import numpy as np
import pynwb
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
    data = EphysMixin()
    data.nwbfile, data.df_name = nwbfile, 'test'
    data.selected_channels = np.arange(len(positions))
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


########################################################
#   restriction to a brain region
########################################################
#   20 electrodes, 20um apart (rel_y), rows 0-9: "SUB", rows 10-19: "VISp" (top)
#   LFP and MUA on the even rows, values = row of the electrode
#   4 units with main channels 15, 3, 12, 8 -> units 0 and 2 in VISp

N_ELECTRODES, MAIN = 20, np.array([15, 3, 12, 8])


class FakeData(EphysMixin):
    """ the ephys methods of physion.analysis.read_NWB.Data on an NWBFile """
    def __init__(self, nwbfile):
        self.nwbfile, self.df_name, self.tlim = nwbfile, 'test', [0., 1.]
        self.select_all_channels()


def ephys_nwbfile():
    from pynwb.ecephys import ElectricalSeries, FeatureExtraction
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
            session_start_time=datetime.datetime.now(datetime.timezone.utc))
    device = nwbfile.create_device(name='probe')
    group = nwbfile.create_electrode_group(name='shank', description='',
                                           location='unknown', device=device)
    for i in range(N_ELECTRODES):
        nwbfile.add_electrode(location=('VISp' if i>=10 else 'SUB'),
                              group=group, rel_x=0., rel_y=20.*i)
    rows = np.arange(0, N_ELECTRODES, 2)
    t = np.arange(100)/100.
    for key in ['LFP', 'MUA']:
        module = nwbfile.create_processing_module(name=key, description='')
        module.add(ElectricalSeries(name=key, timestamps=t,
            data=np.ones((len(t), 1))*rows[np.newaxis,:],
            electrodes=nwbfile.create_electrode_table_region(list(rows), '')))
    T = 30
    waveforms = 0.01*np.ones((T, N_ELECTRODES, len(MAIN)))
    for u, c in enumerate(MAIN):
        waveforms[:, c, u] = np.sin(np.arange(T)/3.)
        nwbfile.add_unit(spike_times=[0.1*(u+1)], electrode_group=group)
    module = nwbfile.create_processing_module(name='Spiking', description='')
    module.add(FeatureExtraction(name='single-unit Waveforms',
        electrodes=nwbfile.create_electrode_table_region(list(range(N_ELECTRODES)), ''),
        description=['unit %i' % u for u in range(len(MAIN))],
        times=np.arange(T)/30e3, features=waveforms))
    return nwbfile


def test_all_channels_by_default():
    data = FakeData(ephys_nwbfile())
    data.build_LFP()
    data.build_depth_units()
    np.testing.assert_array_equal(data.selected_channels, np.arange(N_ELECTRODES))
    np.testing.assert_array_equal(data.LFP[:,0], np.arange(0, N_ELECTRODES, 2))
    np.testing.assert_array_equal(data.main_channel_of_units, MAIN)
    np.testing.assert_array_equal(data.depth_units, 380.-20.*MAIN)


def test_restrict_to_region():
    data = FakeData(ephys_nwbfile())
    for key in ['LFP', 'MUA', 'spikes', 'firing', 'spikeWaveforms', 'depth_units']:
        getattr(data, 'build_%s' % key)()
    data.restrict_to_region('VISp')
    np.testing.assert_array_equal(data.selected_channels, np.arange(10, 20))
    np.testing.assert_array_equal(data.selected_units, [0, 2])
    # LFP, MUA: channels of the region, depth from the top electrode of the region
    for key in ['LFP', 'MUA']:
        np.testing.assert_array_equal(getattr(data, key)[:,0], [10, 12, 14, 16, 18])
        np.testing.assert_array_equal(getattr(data, 'depth_%s' % key),
                                      [180., 140., 100., 60., 20.])
    # units of the region
    assert data.spikes.shape[0] == data.firing.shape[0] == 2
    np.testing.assert_array_equal(np.flatnonzero(data.spikes[0]), [100])  # 0.1s, dt=1ms
    np.testing.assert_array_equal(np.flatnonzero(data.spikes[1]), [300])  # unit 2
    assert data.spikeWaveforms.shape == (30, N_ELECTRODES, 2)
    np.testing.assert_array_equal(data.main_channel_of_units, [15, 12])
    np.testing.assert_array_equal(data.depth_units, [80., 140.])
    # deeper region: depth from its own top electrode (row 9)
    data.restrict_to_region('SUB')
    np.testing.assert_array_equal(data.selected_units, [1, 3])
    np.testing.assert_array_equal(data.depth_LFP, [180., 140., 100., 60., 20.])
    np.testing.assert_array_equal(data.depth_units, [120., 20.])
    # back to all channels
    data.restrict_to_region(None)
    assert data.LFP.shape[0] == 10 and data.spikes.shape[0] == 4


def test_restrict_to_unknown_region():
    data = FakeData(ephys_nwbfile())
    with pytest.raises(ValueError, match='available: SUB, VISp'):
        data.restrict_to_region('CA1')
