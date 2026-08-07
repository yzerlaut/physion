import sys, os, shutil
import spikeinterface.full as si
import spikeinterface.sorters as ss
from physion.assembling.dataset import read_spreadsheet

def run_spike_sorting(self, 
                      datatable=None):
    """   
    """
    if not os.path.isfile(datatable):
        datatable = self.DataTable_file

    print()
    print(' launching spike sorting for the %s ' % datatable)
    print()

    dataset, _, _ = read_spreadsheet(datatable)
    folder  = os.path.dirname(datatable)

    # --- extract information from the DataTable :
    #    (we extract them from the first protocol, they should be fixed for all)
    rec_name = dataset['Npx-Folder'][0] # the OpenEphys recording folder
    electrode_range = dataset['electrode-range'][0]
    bad_channels = dataset['bad-channels'][0]
    npx_path = os.path.join(folder, rec_name)

    print("Reading Open Ephys from:", npx_path)

    # Get Ephys Folder
    rec = si.read_openephys(
        npx_path,
        stream_name='Record Node 101#OneBox-100.ProbeA'
    )

    print("         -> restricting to electrode range %s [...]" % electrode_range)
    e0, e1 = [int(e) for e in electrode_range.split('-')]
    rec = rec.select_channels(rec.get_channel_ids()[e0:e1])

    print("         -> removing n=%i bad channels [...]" % len(bad_channels.split(',')))
    rec = rec.remove_channels(bad_channels.split(','))

    print(" Final number of selected channels:", rec.get_num_channels())

    # Run Sorter
    ks_folder=os.path.join(folder, 'kilosort4_output')
    if os.path.isdir(ks_folder):

        if hasattr(self, 'delBox') and self.delBox.isChecked():
            y = 'yes'
        else:
            y = input(' folder "%s" already exists ! \n Do you want to delete it ? y/[n]' % ks_folder)

        if y in ['y', 'yes']:
            shutil.rmtree(ks_folder)

    if not os.path.isdir(ks_folder):
        print()
        print(' Launching Spike Sorting (through spikeinterface)')
        print()
        sorting = ss.run_sorter(sorter_name='kilosort4', 
                                recording=rec,
                                verbose=True,
                                folder=ks_folder,
                                delete_recording_dat=False)
    else:
        print()
        print(' folder "%s" still exists ! ' % ks_folder)
        print('             remove it manually... ')
        print(' == spike sorting *NOT* launched == ')



if __name__=='__main__':

    run_spike_sorting(None, 
                      datatable=sys.argv[-1])