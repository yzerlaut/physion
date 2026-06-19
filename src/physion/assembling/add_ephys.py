import tempfile, os
import numpy as np
import pandas as pd
from scipy import signal

from spikeinterface.extractors import read_openephys
from spikeinterface.sortingcomponents import peak_detection
from spikeinterface import preprocessing
import spikeinterface.full as si
from pynwb.ecephys import (
    ElectricalSeries,
    FeatureExtraction,
    SpikeEventSeries,
)

from physion.ephys.tools import compute_freq_envelope


def build_args_for_ephys(args, dataset, i, directory):
    args.NPX_folder = os.path.join(directory, dataset['Npx-Folder'][i])
    args.NPX_rec = dataset['Npx-Rec'][i]
    args.Location = dataset['Location'][i]
    args.LFP, args.MUA, args.Spikes = dataset['LFP'][i], dataset['MUA'][i], dataset['Spikes'][i]
    args.raw_Ephys = dataset['raw-Ephys'][i]
    args.electrode_range, args.electrode_subsampling = dataset['electrode-range'][i], dataset['electrode-subsampling'][i]
    args.nStart, args.nStop = dataset['nStart'][i], dataset['nStop'][i]
    # to update, hard-coded for now ...
    args.stream_name='Record Node 101#OneBox-100.ProbeA' 
    args.kilosort_folder = os.path.join(args.NPX_folder, 
            'Record Node 101', 'experiment1', 'recording1', 
            'continuous', 'OneBox-100.ProbeA', 'kilosort4')


def read_kilosort(df):
    """ """
    data = {}
    for key in [f for f in os.listdir(df) if '.npy' in f]:
        data[key.replace('.npy','')] = np.load(os.path.join(df, key), allow_pickle=True)
    for key in [f for f in os.listdir(df) if '.tsv' in f]:
        rd = pd.read_csv(open(os.path.join(df, key)), sep = '\t')
        keys = list(rd.keys())
        for k in keys:
            if k!='cluster_id':
                data[key.replace('.tsv','')+'_'+k] = rd[k]
    return data

def add_ephys(nwbfile, args,
            metadata=None,
            LFP_BAND = [0.5, 300.0],
            MUA_BAND = [300.0, 6000.0],
            resampling_factor = 24, # int,  gives a resampled_rate = 1250,
            margin_ms = 10000,
            chunking_window = '60s'):
    """
    See:
    https://pynwb.readthedocs.io/en/dev/tutorials/domain/ecephys.html
    """

    #   create the device 
    device = nwbfile.create_device(
                        name="Neuropixels OneBox",
                        description="Neuropixels 2.0 probes with OneBox System, creating the folder **%s**" % args.NPX_folder,
                        manufacturer='imec',
                    )

    #   load the open-ephys data:
    # - session
    # session = OpenEphysSession(args.NPX_folder)
    # # - recording node
    # node = int(args.NPX_rec.split('node')[1].split('/')[0])
    # rec_id = int(args.NPX_rec.split('rec')[1])-1
    # rec = session.recordnodes[node].recordings[rec_id]
    # [!!] for later
    #       extract stream_name from "rec" ? --> look at
    #               rec.info['continuous']   

    siRec = read_openephys(args.NPX_folder,
                           stream_name=args.stream_name)

    probes = siRec.get_annotation('probes_info')

    #       [!!] for later:
    # for probe in probes: 
    # rec = rec.set_probe(probe, group_mode="by_shank")
    probe = probes[0]

    # restrict to protocol
    siRec = siRec.frame_slice(start_frame=args.nStart, 
                              end_frame=args.nStop)

    if not hasattr(args, 'tstop_NIdaq'):
        print()
        print(50*'-')
        print(' [!!]  no NIdaq tstop value available ... ')
        print('         --> can not put the proper timestamps of the data')
        print('                     (so putting non-sense)    ')
        print(50*'-')
        print()
        timestamps = np.arange(args.nStop-args.nStart)
    else:
        timestamps = np.linspace(0, args.tstop_NIdaq,
                                 args.nStop-args.nStart)

    # 1)
    # ── remove bad channels ───────────────────────────────────────────
    #
    print("         -> removing bad channels [...]")
    bad_channel_ids, chan_labels = si.detect_bad_channels(siRec,\
                                             method="coherence+psd")
    siRec = siRec.remove_channels(bad_channel_ids)
    good_channels = (chan_labels=='good')

    # 2)
    # ── build Electrode table ───────────────────────────────────────────────
    # 
    print("         -> building electrode table [...]")
    channel_ids = siRec.get_channel_ids()
    locations = siRec.get_property('contact_vector')

    electrode_group = nwbfile.create_electrode_group(
        name        = probe['model_name'],
        description = probe['description'],
        location    = args.Location, # from the DataTable
        device      = device,
    )
    # NWB requires x, y, z; Neuropixels provides x (horizontal) and y (depth).
    # We set z = 0 for a single-shank probe.
    for i in range(len(channel_ids)):

        x = float(locations["x"][i]) if locations is not None else 0.0
        y = float(locations["y"][i]) if locations is not None else float(i) * 25.0

        nwbfile.add_electrode(
            x             = x,
            y             = y,
            z             = 0.0,
            location      = args.Location,
            group         = electrode_group,
        )
    all_electrodes = nwbfile.create_electrode_table_region(
        region      = list(range(len(channel_ids))),
        description = "All electrodes (bad channels removed)",
    )

    # 3)
    #######################################################
    # ── add Spikes ───────────────────────────────────────
    #######################################################
    if args.Spikes=='Yes':

        spiking_module = nwbfile.create_processing_module(
            name        = "Spiking",
            description = "Single Unit Module ",
        )

        sorting = read_kilosort(args.kilosort_folder)
        cInfo = pd.read_csv(\
              open(os.path.join(args.kilosort_folder,
                                'cluster_info.tsv')), sep = '\t')

        template_ids = np.sort(\
            np.unique(sorting['spike_clusters']))

        labels = cInfo['group'] # those are the units manually curated in kilosort
        templates = []

        print("         -> writing single-unit spike times [...]")
        # -------------------------------- #
        #          Spike times             #
        # -------------------------------- #
        def find_matching_unit(id):
            cond = (sorting['spike_clusters']==id)
            return np.unique(sorting['spike_templates'][cond])[0]

        # only units that have been selected as "good" after manual sorting
        for unit_id in template_ids[labels=='good']:

            # getting indices from kilosort
            spike_time_indices = sorting['spike_times'][\
                            sorting['spike_clusters']==unit_id]
            cond = (spike_time_indices>args.nStart) &\
                        (spike_time_indices<args.nStop)

            # we translate those into spike times
            spike_times = [timestamps[s-args.nStart]\
                            for s in spike_time_indices[cond]]
            # we now add to the NWB file
            nwbfile.add_unit(spike_times=spike_times,
                             electrode_group=electrode_group)

            # we store its spike template
            templates.append(sorting['templates'][\
                        find_matching_unit(unit_id)][:,good_channels])

        # -------------------------------- #
        #          Spike templates         #
        # -------------------------------- #
        print("         -> writing single-unit spiking template [...]")
        templates = np.array(templates)

        # features should be --> time, channel, features
        #       and templates is (id, time, channel)
        spike_waveforms = FeatureExtraction(
            name="single-unit Waveforms",
            electrodes=all_electrodes,
            description=['cluster #%i' for i in template_ids[labels=='good']],
            times=np.arange(templates.shape[1])/30e3,
            features=np.array([
                [templates[:,i,k] for k in np.arange(templates.shape[2])]\
                    for i in range(templates.shape[1])])
            )
        spiking_module.add(spike_waveforms)


    ##### FROM NOW ON --> sub-selection of channels ####
    if (args.LFP=='Yes') or (args.MUA=='Yes'):

        print("         -> subsampling channels for voltage traces [...]")

        # channel subsampling
        e0, e1 = [int(e) for e in args.electrode_range.split('-')]
        eSubsampling = np.arange(len(channel_ids))[e0:e1][::int(args.electrode_subsampling)]
        electrodes = nwbfile.create_electrode_table_region(
            region      = list(eSubsampling),
            description = "Chosen electrodes in the range %s with subsampling %s" %\
                    (args.electrode_range, args.electrode_subsampling),
        )
        n_channels = len(eSubsampling)

        # resampling rate for those
        resample_rate = int(siRec.get_sampling_frequency()\
                                    /resampling_factor)

    # 4)
    #######################################################
    # ── add Multi-Unit Activity ──────────────────────────
    #######################################################
    if args.MUA=='Yes':

        print("         -> computing and writing Multi-Unit Activity [...]")

        hfRec = si.resample(
                    si.rectify(
                        si.bandpass_filter(\
                                siRec.select_channels(\
                                    channel_ids =\
                                            siRec.get_channel_ids()[e0:e1]
                                    ), 
                                freq_min=MUA_BAND[0], 
                                freq_max=MUA_BAND[1])
                        ),
                resample_rate=resample_rate
                )


        nTot = n_channels*args.electrode_subsampling # N considered channels
        # compute mean traces 
        mua_traces = hfRec.get_traces()[:,:nTot].reshape(\
            -1, n_channels, args.electrode_subsampling).mean(axis=-1)

        # ── Build NWB MUA objects ───────────────────────────────────────
        mua_es = ElectricalSeries(
            name          = "MUA",
            data          = mua_traces,
            electrodes    = electrodes,
            timestamps    = timestamps[::resampling_factor][:mua_traces.shape[0]],
            conversion    = 1e-6,   # µV → V
            description   = (
                f"MUA signal in uV "
                f"electrode channels : {args.electrode_range}"
                f"electrode subsampling: {args.electrode_subsampling}"
                f"MUA band ({MUA_BAND[0]}–{MUA_BAND[1]} Hz, "
                f"Butterworth order 5, zero-phase), "
                f"downsampled to {resample_rate} Hz. "
            ),
        )
    
        mua_module = nwbfile.create_processing_module(
            name        = "MUA",
            description = "Multi-Unit-Activity computed from raw electrophysiology data",
        )
        mua_module.add(mua_es)


    # 5)
    #######################################################
    # ── add Local Field Potential  ───────────────────────
    #######################################################
    if args.LFP=='Yes':

        print("         -> computing and writing LFP band [...]")

        # subsampling on the chosen electrodes
        siRec = siRec.select_channels(
            channel_ids = siRec.get_channel_ids()[eSubsampling]
        ) 

        temp_folder = os.path.join(tempfile.gettempprefix(), 'temp')
        # ── 1. We save the data in the memory with an **extended** chunk size to avoid boundary artefacts
        if True: 
            siRec.save(format='binary', 
                        folder=temp_folder, 
                        chunk_duration=chunking_window,
                        overwrite=True,
                        n_jobs=0.8, #
                        progress_bar=True)

        rec = si.load(temp_folder,
                    chunk_duration=chunking_window)

        # ── 2. Apply filter + resample pipeline on the extended chunk ─────
        rec_lfp = si.resample(
                si.bandpass_filter(rec, 
                    freq_min=LFP_BAND[0], 
                    freq_max=LFP_BAND[1],
                    ignore_low_freq_error=True,
                    margin_ms=margin_ms
                ),
                resample_rate=resample_rate)

        # ── 3. Build NWB LFP objects ───────────────────────────────────────
        lfp_es = ElectricalSeries(
            name          = "LFP",
            data          = rec_lfp.get_traces(),
            electrodes    = electrodes,
            timestamps    = timestamps[::resampling_factor][:rec_lfp.get_num_frames()],
            conversion    = 1e-6,   # µV → V
            description   = (
                f"LFP signal in uV "
                f"electrode channels : {args.electrode_range}"
                f"electrode subsampling: {args.electrode_subsampling}"
                f"LFP band ({LFP_BAND[0]}–{LFP_BAND[1]} Hz, "
                f"Butterworth order 5, zero-phase), "
                f"downsampled to {resample_rate} Hz. "
                f"Chunk margin: {margin_ms} ms per side, Chunking window: {chunking_window}"
            ),
        )
    
        lfp_module = nwbfile.create_processing_module(
            name        = "LFP",
            description = "Local-Field Potential computed from raw electrophysiology data",
        )
        lfp_module.add(lfp_es)

if __name__=='__main__':

    print('test')
