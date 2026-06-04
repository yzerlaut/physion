import tempfile, os
import numpy as np
import pandas as pd

# from physion.ephys.tools import filtering, 
from spikeinterface.extractors import read_openephys
from spikeinterface.sortingcomponents import peak_detection
import spikeinterface.full as si
from pynwb.ecephys import (
    ElectricalSeries,
    FeatureExtraction,
    SpikeEventSeries,
)

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
            lfp_freq_min = 0.5,
            lfp_freq_max = 300.0,
            mua_freq_min = 300.0,
            mua_freq_max = 3000.0,
            resample_rate = 1250,
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
    # print("         -> removing bad channels [...]")
    # bad_channel_ids, chan_labels = si.detect_bad_channels(siRec, method="coherence+psd")
    # siRec = siRec.remove_channels(bad_channel_ids)
    # good_channels = (chan_labels=='good')

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
        description = "All electrodes",
    )

    if args.Spikes=='Yes':

        spiking_module = nwbfile.create_processing_module(
            name        = "Spiking",
            description = "Single and Multiple Unit Module ",
        )

        # sorting = si.read_kilosort(folder)

        sorting = read_kilosort(args.kilosort_folder)
        template_ids = np.sort(\
            np.unique(sorting['spike_templates']))
        labels = sorting['cluster_group_KSLabel']
        templates = []
        # -------------------------------- #
        #          Single Units            #
        # -------------------------------- #
        print("         -> writing single-unit spike times [...]")
        for unit_id in template_ids[labels=='good']:

            # # --- old
            # spike_time_indices = sorting.get_unit_spike_train(unit_id,
            #                                 start_frame=args.nStart,
            #                                 end_frame=args.nStop)

            spike_time_indices = sorting['spike_times'][\
                            sorting['spike_templates']==unit_id]
            cond = (spike_time_indices>args.nStart) &\
                        (spike_time_indices<args.nStop)

            # we translate those into spike times
            spike_times = [timestamps[s-args.nStart]\
                            for s in spike_time_indices[cond]]
            # we now add to the NWB file
            nwbfile.add_unit(spike_times=spike_times, 
                             electrode_group=electrode_group)

            # we store its spike template
            templates.append(sorting['templates'][unit_id]) 
        # -------------------------------- #
        #          Spike templates         #
        # -------------------------------- #
        print("         -> writing single-unit spiking template [...]")
        # [!!] for now still need to read the kilosort data
        #       the spike-interface layer doesn't work to extract templates
        # templates = np.load(os.path.join(folder, 'templates.npy'), 
        #                     allow_pickle=True)
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

        # -------------------------------- #
        #        multi-unit Events         #
        # -------------------------------- #
        # print("         -> running peak-detection (locally_exclusive) [...]")
        # peaks = peak_detection.detect_peaks(siRec,
        #         method='locally_exclusive',
        #         method_kwargs=dict(peak_sign='both'),
        #                     )

        times, channels = [], []
        for unit_id in template_ids[labels=='mua']:
            cond = (sorting['spike_times']>args.nStart) &\
                     (sorting['spike_times']<args.nStop)
            times += list(\
                sorting['spike_times'][\
                 cond & (sorting['spike_clusters']==unit_id)])
            # to find the associated channel, we use the template
            channel = \
                np.argmax(sorting['templates'][unit_id].std(axis=0))            
            channels += list(\
                channel*np.ones(np.sum(cond & (sorting['spike_clusters']==unit_id))))
            
        muEvents = SpikeEventSeries(
            name="multi-unit Events",
            data = channels,
            electrodes=all_electrodes,
            timestamps= [timestamps[t-args.nStart] for t in times])

        spiking_module.add(muEvents)

    # -------------------------------------------------#
    # 
    ##### FROM NOW ON --> sub-selection of channels ####
    #
    # -------------------------------------------------#
    if (args.raw_Ephys=='Yes') or (args.LFP=='Yes') or (args.MUA=='Yes'):
        print("         -> subsampling channels for voltage traces [...]")
        # subsampling
        e0, e1 = [int(e) for e in args.electrode_range.split('-')]
        eSubsampling = np.arange(len(channel_ids))[e0:e1][::int(args.electrode_subsampling)]
        electrodes = nwbfile.create_electrode_table_region(
            region      = list(eSubsampling),
            description = "Chosen electrodes in the range %s with subsampling %s" %\
                    (args.electrode_range, args.electrode_subsampling),
        )
        n_channels = len(eSubsampling)

        siRec = siRec.select_channels(
            channel_ids = siRec.get_channel_ids()[eSubsampling]
        ) 

        temp_folder = os.path.join(tempfile.gettempprefix(), 'temp')
        if True: # TURN BACK TO TRUE AFTER DEBUGGING !!
            # # we save the data in the memory with an **extended** chunk size
            siRec.save(format='binary', 
                        folder=temp_folder, overwrite=True,
                            chunk_duration=chunking_window,
                                n_jobs=0.8, #
                                    progress_bar=True)

        rec = si.load(temp_folder,
                    chunk_duration=chunking_window)
    else:
        rec = None

    # ── 3e. Raw AP band ───────────────────────────────────────────────────
    if args.raw_Ephys=='Yes' and (rec is not None):

        print("         - Writing raw EPhys data [...]")

        raw_es = ElectricalSeries(
            name             = "ElectricalSeries",
            data             = rec.get_traces(),
            electrodes       = electrodes,
            timestamps       = timestamps,
            conversion       = 1e-6,   # µV → V  (NWB stores Volts)
            description      = "Raw Electrophysiological Data",
            comments         = (
                f"Open-Ephys recording, selecting {n_channels} channels, "
                f"electrode channels : {args.electrode_range}, "
                f"electrode subsampling: {args.electrode_subsampling}"
            )
        )
        nwbfile.add_acquisition(raw_es)


    # ──  LFP band ──────────────────────────────────────────────────────
    if args.LFP=='Yes' and (rec is not None):

        print("         - Computing and writing LFP band [...]")

        # ── 2. Apply filter + resample pipeline on the extended chunk ─────
        rec_lfp = si.bandpass_filter(rec, 
            freq_min=lfp_freq_min, freq_max=lfp_freq_max,
            ignore_low_freq_error=True,
            margin_ms=margin_ms
        )
        resampling_factor = int(rec.get_sampling_frequency()/resample_rate)
        rec_lfp = si.resample(rec_lfp, 
                resample_rate=int(rec.get_sampling_frequency()/resampling_factor))

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
                f"LFP band ({lfp_freq_min}–{lfp_freq_max} Hz, "
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

    # ──  MUA band ──────────────────────────────────────────────────────
    if False:

        print("         -> Computing and writing Multi-Unit Activity [...]")
        print()
        print("                      [!!] TODO: need to take absolute value and sliding mean ")
        print()

        # ── 2. Apply filter + resample pipeline on the extended chunk ─────
        rec_mua = si.bandpass_filter(
            rec, 
            freq_min=mua_freq_min, freq_max=mua_freq_max,
            ignore_low_freq_error=True,
            margin_ms=3000,
        )
        rec_mua = si.resample(rec_mua, 
                            resample_rate=resample_rate)

        # ── 3. Build NWB MUA objects ───────────────────────────────────────
        mua_es = ElectricalSeries(
            name          = "MUA",
            data          = rec_mua.get_traces(),
            electrodes    = electrodes,
            timestamps    = timestamps[::resampling_factor][:rec_mua.get_num_frames()],
            conversion    = 1e-6,   # µV → V
            description   = (
                f"MUA signal in uV "
                f"electrode channels : {args.electrode_range}"
                f"electrode subsampling: {args.electrode_subsampling}"
                f"MUA band ({mua_freq_min}–{mua_freq_max} Hz, "
                f"Butterworth order 5, zero-phase), "
                f"downsampled to {resample_rate} Hz. "
                f"Chunk margin: {margin_ms} ms per side, Chunking window: {chunking_window}"
            ),
        )
    
        mua_module = nwbfile.create_processing_module(
            name        = "MUA",
            description = "Multi-Unit-Activity computed from raw electrophysiology data",
        )
        mua_module.add(mua_es)


if __name__=='__main__':

    print('test')
