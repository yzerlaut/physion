"""
electrode table and landmarks of the ephys NWB files (assembling/add_ephys.py)
    written at the assembling, modified in place afterwards
"""
import datetime, types
import numpy as np
import pynwb
from pynwb.ecephys import ElectricalSeries

from physion.assembling.add_ephys import add_electrode_table, add_landmarks_table
from physion.dataviz.ephys import contact_position

PROBE = {'model_name':'NP2014', 'description':'Neuropixels 2.0'}


def build(fn, probe_channels=np.arange(10, 20)):
    """ 10 kept channels (e.g. electrode-range 10-20) + a subsampled LFP """
    nwbfile = pynwb.NWBFile(session_description='test', identifier='test',
            session_start_time=datetime.datetime.now(datetime.timezone.utc))
    device = nwbfile.create_device(name='probe')
    channel_ids = ['CH%i' % c for c in probe_channels]
    positions = {'x':np.zeros(len(probe_channels)), 'y':15.*probe_channels}
    add_electrode_table(nwbfile, device, PROBE, channel_ids, probe_channels,
                        contact_positions=positions)
    add_landmarks_table(nwbfile)
    lfp_elecs = nwbfile.create_electrode_table_region(
            region=list(range(0, len(probe_channels), 5)), description='subsampled')
    nwbfile.add_acquisition(ElectricalSeries(name='LFP', data=np.zeros((10, 2)),
                                             electrodes=lfp_elecs, rate=10.))
    with pynwb.NWBHDF5IO(fn, 'w') as io:
        io.write(nwbfile)


def test_electrode_table(tmp_path):
    fn = str(tmp_path/'test.nwb')
    build(fn)
    with pynwb.NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        elecs = nwbfile.electrodes
        # no brain coordinates, positions on the probe
        assert ('x' not in elecs.colnames) and ('rel_y' in elecs.colnames)
        np.testing.assert_array_equal(elecs['probe_channel'][:], np.arange(10, 20))
        np.testing.assert_allclose(elecs['rel_y'][:], 15.*np.arange(10, 20))
        assert list(elecs['channel_name'][:])[0] == 'CH10'
        # brain regions: written after the assembling
        assert set(elecs['location'][:]) == {'unknown'}
        # LFP rows -> probe channels
        np.testing.assert_array_equal(
            nwbfile.acquisition['LFP'].electrodes[:]['probe_channel'], [10, 15])
        # landmarks: L4 not determined yet
        landmarks = nwbfile.processing['Landmarks']['Landmarks']
        assert list(landmarks['landmark'][:]) == ['L4']
        assert landmarks['channel'][0] == -1


def test_modified_after_assembling(tmp_path):
    """ what the location script/GUI will do: in place modification """
    fn = str(tmp_path/'test.nwb')
    build(fn)
    with pynwb.NWBHDF5IO(fn, 'a') as io:
        nwbfile = io.read()
        probe_channels = nwbfile.electrodes['probe_channel'][:]
        location = nwbfile.electrodes['location'].data
        for i, c in enumerate(probe_channels):
            location[i] = 'SUB' if c<15 else 'VISp'
        landmarks = nwbfile.processing['Landmarks']['Landmarks']
        landmarks['channel'].data[0] = 17
        landmarks['method'].data[0] = 'CSD'
    with pynwb.NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        assert list(nwbfile.electrodes['location'][:]) == 5*['SUB']+5*['VISp']
        assert list(nwbfile.acquisition['LFP'].electrodes[:]['location']) == ['SUB', 'VISp']
        landmarks = nwbfile.processing['Landmarks']['Landmarks']
        assert (landmarks['channel'][0], landmarks['method'][0]) == (17, 'CSD')


def test_contact_position(tmp_path):
    fn = str(tmp_path/'test.nwb')
    build(fn)
    with pynwb.NWBHDF5IO(fn, 'r') as io:
        data = types.SimpleNamespace(nwbfile=io.read())
        assert contact_position(data, 2) == (0., 15.*12)
