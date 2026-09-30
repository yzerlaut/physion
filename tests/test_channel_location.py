"""
location of the ephys channels written from the DataTable (ephys/location_gui.py)
"""
import numpy as np
import pandas as pd
import pynwb
import pytest

from physion.ephys.location_gui import add_locations_from_DataTable,\
        parse_channel_range, parse_channel, electrode_locations
from test_electrode_table import build # NWB file with probe channels 10-19


def write_DataTable(fn, rows):
    """ "Recordings" sheet (+ the other sheets of the DataTables) """
    with pd.ExcelWriter(fn) as writer:
        pd.DataFrame(rows).to_excel(writer, sheet_name='Recordings', index=False)
        pd.DataFrame({'subject':['S1']}).to_excel(writer, sheet_name='Subjects', index=False)


def read(fn):
    with pynwb.NWBHDF5IO(fn, 'r') as io:
        nwbfile = io.read()
        table = nwbfile.processing['Landmarks']['Landmarks']
        return {'location':list(nwbfile.electrodes['location'][:]),
                'group':nwbfile.electrode_groups['NP2014'].location,
                'L4':table['channel'][0], 'method':table['method'][0]}


def test_parsing():
    assert parse_channel_range('100-200') == (100, 200)
    assert parse_channel_range(np.nan) is None
    assert parse_channel(150.0) == 150 and parse_channel('') is None
    with pytest.raises(ValueError):
        parse_channel_range('200-100')
    with pytest.raises(ValueError): # overlapping regions
        electrode_locations(np.arange(10), {'SUB':(0, 6), 'VISp':(5, 10)})


def test_locations_from_DataTable(tmp_path):
    (tmp_path/'NWBs').mkdir()
    build(str(tmp_path/'NWBs'/'2026_08_20-16-36-00.nwb'))
    build(str(tmp_path/'NWBs'/'2026_08_20-17-00-00.nwb'))
    table = str(tmp_path/'DataTable.xlsx')
    write_DataTable(table, {
        'day':['2026_08_20']*3, 'time':['16-36-00', '17-00-00', '18-00-00'],
        'SUB-channels':['0-14', '0-12', np.nan],
        'VISp-channels':['14-100', '12-15', np.nan],
        'L4-center-channel':[17, np.nan, np.nan],
        'WRONG-channels':['', '', '']})

    written = add_locations_from_DataTable(table) # default: the "NWBs" folder
    assert len(written) == 2 # no NWB file for the third recording

    first = read(written[0])
    assert first['location'] == 4*['SUB']+6*['VISp']
    assert first['group'] == 'SUB, VISp'
    assert first['L4'] == 17 and 'DataTable' in first['method']

    second = read(written[1])
    assert second['location'] == 2*['SUB']+3*['VISp']+5*['unknown']
    assert second['L4'] == -1

    # DataTable modified -> NWB files updated (the DataTable is the reference)
    write_DataTable(table, {
        'day':['2026_08_20'], 'time':['16-36-00'],
        'VISp-channels':['10-20'], 'L4-center-channel':[np.nan]})
    add_locations_from_DataTable(table, str(tmp_path/'NWBs'))
    first = read(written[0])
    assert first['location'] == 10*['VISp']
    assert first['group'] == 'VISp' and first['L4'] == -1


def test_not_written_for_overlapping_regions(tmp_path):
    build(str(tmp_path/'2026_08_20-16-36-00.nwb'))
    table = str(tmp_path/'DataTable.xlsx')
    write_DataTable(table, {'day':['2026_08_20'], 'time':['16-36-00'],
                            'SUB-channels':['0-15'], 'VISp-channels':['14-20']})
    assert add_locations_from_DataTable(table, str(tmp_path)) == []
    assert read(str(tmp_path/'2026_08_20-16-36-00.nwb'))['location'] == 10*['unknown']
