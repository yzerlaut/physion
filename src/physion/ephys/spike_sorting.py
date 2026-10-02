# %%
import sys, os, shutil
import numpy as np
import pandas as pd

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

    if type(bad_channels) in [str, np.str_]:
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

# %%

def read_kilosort_output(df):
    """ 
    """
    data = {}

    # ---    raw kilosort output   --- #
    for key in [f for f in os.listdir(df) if '.npy' in f]:
        data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)

    # ---  tsv files edited by Phy --- #
    for key in [f for f in os.listdir(df) if '.tsv' in f]:
        rd = pd.read_csv(open(os.path.join(df, key)), sep = '\t')
        keys = list(rd.keys())
        for k in keys:
            data[key.replace('.tsv','')+'_'+k] = rd[k]

    # ---    phy output (useless) --- #
    # if os.path.isdir(os.path.join(df, '.phy')):
    #     with open(os.path.join(df, '.phy', 'new_cluster_id.json')) as f:
    #         data['new_cluster_id'] = json.load(f)
    #     with open(os.path.join(df, '.phy', 'state.json')) as f:
    #         data['phy_state'] = json.load(f)

    return data

def find_template_of_cluster(cluster_id, data):
    """ 
    using:
        spike_templates.npy --> template ID for every detected spike (not rewritten by Phy)
        spike_clusters.npy  --> cluster ID for every detected spike (**rewritten** by Phy)
    """
    values =  np.unique(
        data['spike_templates'][\
            data['spike_clusters']==cluster_id])
    if len(values)==1:
        return values[0]
    else:
        print()
        print(' [!!]  cluster %i does not have a single matching template: %s ' % (cluster_id, values))
        print()


def fetch_good_units(data):

    # units manually selected as good in Phy:
    good_units = data['cluster_info_group']=='good'

    cluster_ids = data['cluster_info_cluster_id'][good_units]

    spike_time_indices, templates = [], []
    
    for cluster_id in cluster_ids:

        # getting indices from kilosort:
        spk_time_indices = data['spike_times'][\
                        data['spike_clusters']==cluster_id]

        # we find its spike template:
        template_id = find_template_of_cluster(cluster_id, data)

        # put in storage lists:
        spike_time_indices.append(spk_time_indices)
        templates.append(\
            data['templates'][template_id, :, :])

    templates = np.array(templates)

    return spike_time_indices, templates


if __name__=='__main__':

    run_spike_sorting(None, 
                      datatable=sys.argv[-1])

# %%
if False:
    import sys
    sys.path.append('/home/user/lab-notebook/yann/physion/src')
    from physion.ephys.spike_sorting import read_kilosort_output, fetch_good_units
    data = read_kilosort_output(\
                os.path.join('/media/user/DATA2/2026_08_04/',
                        'kilosort4_output', 'sorter_output'))
    #indices, templates = fetch_good_units(data)
    
    print(find_template_of_cluster(152, data))        
# %%
