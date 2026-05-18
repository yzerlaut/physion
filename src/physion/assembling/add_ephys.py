import numpy as np
import tempfile, os
# from physion.ephys.tools import filtering, 
from spikeinterface.extractors import read_openephys
import spikeinterface.full as si
from pynwb.ecephys import (
    ElectricalSeries,
    ElectrodeGroup,
    LFP,
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
                        description="Neuropixels 2.0 probes with OneBox System",
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
                       stream_name='Record Node 101#OneBox-100.ProbeA')

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

    # ── Electrode table ───────────────────────────────────────────────
    channel_ids = siRec.get_channel_ids()
    locations = siRec.get_property('contact_vector')

    electrode_group = nwbfile.create_electrode_group(
        name        = probe['model_name'],
        description = probe['description'],
        location    = args.Location, # from the DataTable
        device      = device,
    )

    # subsampling
    e0, e1 = [int(e) for e in args.electrode_range.split('-')]
    eSubsampling = np.arange(len(channel_ids))[e0:e1][::int(args.electrode_subsampling)]

    # NWB requires x, y, z; Neuropixels provides x (horizontal) and y (depth).
    # We set z = 0 for a single-shank probe.
    for i in eSubsampling:

        x = float(locations["x"][i]) if locations is not None else 0.0
        y = float(locations["y"][i]) if locations is not None else float(i) * 25.0

        nwbfile.add_electrode(
            x             = x,
            y             = y,
            z             = 0.0,
            location      = args.Location,
            group         = electrode_group,
        )

    electrodes = nwbfile.create_electrode_table_region(
        region      = list(range(len(eSubsampling))),
        description = "Chosen electrodes in the range %s with subsampling %s" %\
                (args.electrode_range, args.electrode_subsampling),
    )
    n_channels = len(eSubsampling)

    siRec = siRec.select_channels(
        channel_ids = siRec.get_channel_ids()[eSubsampling]
    ) 

    # # we save the data in the memory with an **extended** chunk size
    temp_folder = os.path.join(tempfile.gettempprefix(), 'temp')
    siRec.save(format='binary', 
                folder=temp_folder, overwrite=True,
                    chunk_duration=chunking_window,
                        n_jobs=0.8, #
                            progress_bar=True)

    rec = si.load(temp_folder, chunk_duration=chunking_window)

    # ── 3e. Raw AP band ───────────────────────────────────────────────────
    if args.raw_Ephys=='Yes':

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
    if args.LFP=='Yes':

        print("         - Computing and writing LFP band [...]")

        # ── 2. Apply filter + resample pipeline on the extended chunk ─────
        rec_lfp = si.bandpass_filter(
            rec, 
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
        lfp_module.add(LFP(electrical_series=lfp_es))

    # ──  MUA band ──────────────────────────────────────────────────────
    if args.MUA=='Yes':

        print("         - Computing and writing Multi-Unit Activity [...]")
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



def add_spikes(nwbfile, args,
                metadata=None):

    # ── 3g. Spike events (optional) ───────────────────────────────────────
    # Requires a pre-computed spike-sorting result (e.g. from Kilosort).
    # SpikeInterface can load Kilosort / Phy output with:
    #   sorting = si.read_kilosort(phy_folder)
    if False:
        print("Writing spike events … (not implemented in this template)")
        # Example skeleton:
        #
        # sorting = si.read_kilosort(Path("/path/to/phy/output"))
        # for unit_id in sorting.get_unit_ids():
        #     spike_times = sorting.get_unit_spike_train(unit_id, return_times=True)
        #     nwbfile.add_unit(spike_times=spike_times, electrode_group=electrode_group)
 