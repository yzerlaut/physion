"""
Add the location of the ephys channels to the NWB files (after the assembling)

from the DataTable ("Recordings" sheet), one column per recorded region:
    - "<region>-channels" (e.g. "VISp-channels"): the channel range of the
            region on the probe, e.g. "100-200" (channels 100 to 199,
            same convention than the "electrode-range" column)
    - "L4-center-channel": the channel of the L4 landmark
(channels: index on the probe, the "probe_channel" column of the electrodes)

written in the NWB files (in place, see assembling/add_ephys.py):
    - the "location" column of the electrodes ("unknown" outside the regions)
    - the location of the electrode group (regions along the probe)
    - the "L4" row of the "Landmarks" table

usage:
    python -m physion.ephys.location_gui DataTable.xlsx [NWB-folder]
    or in the GUI: "Assembling / Add Channel Location to NWB"
"""
import os, datetime
import numpy as np
import pandas as pd
import h5py, pynwb


allen_brain_regions = {
    "VISp": "Primary visual cortex",
    "VISl": "Lateromedial visual area",
    "VISrl": "Rostrolateral visual area",
    "VISal": "Anterolateral visual area",
    "VISpm": "Posteromedial visual area",
    "VISam": "Anteromedial visual area",
    "CA1": "Cornu ammonis 1",
    "CA3": "Cornu ammonis 3",
    "DG": "Dentate gyrus",
    "SUB": "Subiculum",
    "ProS": "Prosubiculum",
    "LGd": "Dorsal part of the lateral geniculate nucleus",
    "LP": "Lateral posterior nucleus of the thalamus",
    "APN": "Anterior pretectal nucleus",
    "MB": "Midbrain",
    "SCsg": "Superior colliculus, superficial gray layer",
    "RSP": "Retrosplenial cortex",
    "PPT": "Posterior parietal cortex"
}

UNKNOWN = 'unknown' # location outside the regions of the DataTable
L4_KEY = 'L4-center-channel'


def region_key(region):
    return '%s-channels' % region


def is_empty(value):
    return (value is None) or (isinstance(value, float) and np.isnan(value))\
                or (str(value).strip() in ['', 'nan'])


def parse_channel_range(value):
    """ "100-200" -> (100, 200), None if empty """
    if is_empty(value):
        return None
    c0, c1 = [int(float(c)) for c in str(value).strip().split('-')]
    if c1<=c0:
        raise ValueError('channel range "%s": the end should be above the start' % value)
    return c0, c1


def parse_channel(value):
    """ "150" (or 150.0 from excel) -> 150, None if empty """
    if is_empty(value):
        return None
    return int(float(value))


def read_locations(row):
    """
    regions {region: (channel_start, channel_stop)} and L4 channel
        of a row of the DataTable
    """
    regions = {}
    for region in allen_brain_regions:
        if region_key(region) in row:
            channels = parse_channel_range(row[region_key(region)])
            if channels is not None:
                regions[region] = channels
    L4_channel = parse_channel(row[L4_KEY]) if (L4_KEY in row) else None
    return regions, L4_channel


def unknown_region_columns(dataset):
    """ "*-channels" columns that are not in the list of regions """
    return [c for c in dataset.columns if c.endswith('-channels') and\
                    (c.replace('-channels', '') not in allen_brain_regions)]


def electrode_locations(probe_channels, regions):
    """ location of each electrode from the channel ranges of the regions """
    locations = np.full(len(probe_channels), UNKNOWN, dtype=object)
    for region, (c0, c1) in regions.items():
        cond = (probe_channels>=c0) & (probe_channels<c1)
        overlap = cond & (locations!=UNKNOWN)
        if np.sum(overlap)>0:
            raise ValueError('channels %s of "%s" already in "%s"' %\
                    (probe_channels[overlap], region, locations[overlap][0]))
        locations[cond] = region
    return locations


def write_locations(nwb_file, regions, L4_channel=None,
                    method='', verbose=True):
    """
    writes in place the location of the electrodes and the L4 landmark,
        the DataTable is the reference: the electrodes outside the regions
        are "unknown" and L4 is reset (channel=-1) if not given
    """
    with pynwb.NWBHDF5IO(nwb_file, 'a') as io:

        nwbfile = io.read()

        if (nwbfile.electrodes is None) or\
                ('probe_channel' not in nwbfile.electrodes.colnames):
            raise ValueError('no "probe_channel" in the electrodes of "%s"' % nwb_file+\
                    ' (NWB file built before the location support, re-build it)')

        probe_channels = np.array(nwbfile.electrodes['probe_channel'][:])
        locations = electrode_locations(probe_channels, regions)

        # 1) electrodes
        data = nwbfile.electrodes['location'].data
        for i, location in enumerate(locations):
            data[i] = location

        # 2) L4 landmark
        if L4_channel is not None:
            if L4_channel not in probe_channels:
                print('   [!!] L4 channel %i not among the recorded channels' % L4_channel)
            landmark = {'channel':L4_channel, 'method':method,
                        'date':datetime.date.today().strftime('%Y_%m_%d')}
        else:
            landmark = {'channel':-1, 'method':'', 'date':''}
        table = nwbfile.processing['Landmarks']['Landmarks']
        iL4 = list(table['landmark'][:]).index('L4')
        for key in landmark:
            table[key].data[iL4] = landmark[key]

        group_names = list(nwbfile.electrode_groups.keys())

    # 3) location of the electrode groups: regions along the probe
    #       (an attribute, not modifiable with pynwb)
    group_location = ', '.join(sorted(regions, key=lambda r: regions[r][0]))
    with h5py.File(nwb_file, 'a') as f:
        for name in group_names:
            f['general/extracellular_ephys/%s' % name].attrs['location'] =\
                    group_location if group_location!='' else UNKNOWN

    if verbose:
        for region in sorted(regions, key=lambda r: regions[r][0]):
            c0, c1 = regions[region]
            n = np.sum(locations==region)
            print('     - %-6s channels %i-%i: n=%i electrodes %s' % (region, c0, c1, n,
                        '' if n>0 else ' [!!] none of the recorded channels'))
        print('     - L4 landmark: %s' % (L4_channel if L4_channel is not None else 'not set'))

    return locations


def nwb_file_of_row(row, nwb_folder):
    """ NWB file of a recording of the DataTable: "day-time.nwb" """
    return os.path.join(nwb_folder, '%s-%s.nwb' % (row['day'], row['time']))


def add_locations_from_DataTable(DataTable_file, nwb_folder=None, verbose=True):
    """
    loops over the recordings of the DataTable,
        returns the list of NWB files written
    """
    if nwb_folder is None:
        nwb_folder = os.path.join(os.path.dirname(DataTable_file), 'NWBs')

    dataset = pd.read_excel(DataTable_file, sheet_name='Recordings')
    method = 'from "%s" of the DataTable "%s"' % (L4_KEY, os.path.basename(DataTable_file))

    for c in unknown_region_columns(dataset):
        print(' [!!] column "%s": region not in the list of regions (ignored)' % c)

    written = []
    for i, row in dataset.iterrows():

        nwb_file = nwb_file_of_row(row, nwb_folder)
        if not os.path.isfile(nwb_file):
            if verbose:
                print(' [%i] %s  -> no NWB file' % (i+1, os.path.basename(nwb_file)))
            continue

        print(' [%i] %s ' % (i+1, os.path.basename(nwb_file)))
        try:
            regions, L4_channel = read_locations(row)
            write_locations(nwb_file, regions, L4_channel,
                            method=method, verbose=verbose)
            written.append(nwb_file)
        except (ValueError, KeyError, OSError) as e:
            print('     -> [!!] not written: %s' % e)

    print(' -> location written in n=%i NWB files' % len(written))
    return written


#################################################
####            GUI                       #######
#################################################

from PyQt5 import QtWidgets
from physion.utils.paths import FOLDERS
from physion.gui.window import Window


class ChannelLocationWindow(Window):

    name = 'add_channel_location'

    def __init__(self, main, tab_id=1):

        super().__init__(main, tab_id)
        self.DataTable_file, self.nwb_folder = None, None

        # ========================================================
        #------------------- SIDE PANELS FIRST -------------------
        self.add_side_widget(QtWidgets.QLabel(' _-* CHANNEL LOCATION *-_ '))

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.add_side_widget(QtWidgets.QLabel('from:'),
                             spec='small-left')
        self.folderBox = QtWidgets.QComboBox(self.main)
        self.folderBox.addItems(FOLDERS.keys())
        self.add_side_widget(self.folderBox, spec='large-right')

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.DataTableBtn = QtWidgets.QPushButton(' choose DataTable.xlsx ⬇')
        self.DataTableBtn.clicked.connect(self.choose_DataTable)
        self.add_side_widget(self.DataTableBtn)
        self.DataTableLabel = QtWidgets.QLabel('   -')
        self.add_side_widget(self.DataTableLabel)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.nwbFolderBtn = QtWidgets.QPushButton(' choose NWB folder ⬇')
        self.nwbFolderBtn.clicked.connect(self.choose_NWB_folder)
        self.add_side_widget(self.nwbFolderBtn)
        self.nwbFolderLabel = QtWidgets.QLabel('   -')
        self.add_side_widget(self.nwbFolderLabel)

        self.add_side_widget(QtWidgets.QLabel(' '))
        self.add_side_widget(QtWidgets.QLabel(' DataTable columns, e.g.:'))
        self.add_side_widget(QtWidgets.QLabel('   "VISp-channels": 100-200'))
        self.add_side_widget(QtWidgets.QLabel('   "%s": 150' % L4_KEY))
        self.add_side_widget(QtWidgets.QLabel(' '))

        self.runBtn = QtWidgets.QPushButton('  * - RUN - * ')
        self.runBtn.clicked.connect(self.run)
        self.add_side_widget(self.runBtn)

        self.add_side_widget(QtWidgets.QLabel(' '))

        self.refresh_tab()

    def choose_DataTable(self):

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self.main,
                     "Select DataTable (xlsx file) ",
                     self.choose_root_folder(),
                     options=QtWidgets.QFileDialog.DontUseNativeDialog,
                     filter="*.xlsx")

        if filename!='':
            self.DataTable_file = filename
            self.DataTableLabel.setText('   %s' % os.path.basename(filename))
            # default NWB folder: "NWBs" next to the DataTable
            folder = os.path.join(os.path.dirname(filename), 'NWBs')
            if (self.nwb_folder is None) and os.path.isdir(folder):
                self.set_NWB_folder(folder)

    def choose_NWB_folder(self):
        folder = self.open_folder()
        if folder!='':
            self.set_NWB_folder(folder)

    def set_NWB_folder(self, folder):
        self.nwb_folder = folder
        self.nwbFolderLabel.setText('   %s' % os.path.basename(folder))

    def run(self):
        if self.DataTable_file is None:
            print(' ----> no DataTable file selected, choose a valid one ...')
        elif self.nwb_folder is None:
            print(' ----> no NWB folder selected, choose a valid one ...')
        else:
            written = add_locations_from_DataTable(self.DataTable_file, self.nwb_folder)
            self.statusBar.showMessage(' location written in n=%i NWB files' % len(written))


if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("DataTable_file", type=str)
    parser.add_argument("nwb_folder", type=str, nargs='?', default=None,
                        help='default: the "NWBs" folder next to the DataTable')
    args = parser.parse_args()

    add_locations_from_DataTable(args.DataTable_file, args.nwb_folder)
