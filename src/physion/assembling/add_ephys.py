import numpy as np
import tempfile, os
from physion.ephys.LFP.build import write_lfp_to_nwb
from spikeinterface.extractors import read_openephys
import spikeinterface.full as si
from pynwb.ecephys import (
    ElectricalSeries,
    ElectrodeGroup,
    LFP,
    SpikeEventSeries,
)

def add_ephys(nwbfile, args,
            metadata=None,
            electrode_subsampling= 50,
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

    ELECTRODE_LOCATION = 'VIS' # to put in the metadata

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

    channel_ids = siRec.get_channel_ids()
    locations = siRec.get_property('contact_vector')

    electrode_group = nwbfile.create_electrode_group(
        name        = probe['model_name'],
        description = probe['description'],
        location    = ELECTRODE_LOCATION,
        device      = device,
    )
 
    # ── Electrode table ───────────────────────────────────────────────
    # NWB requires x, y, z; Neuropixels provides x (horizontal) and y (depth).
    # We set z = 0 for a single-shank probe.
    for i, ch_id in enumerate(channel_ids):

        x = float(locations["x"][i]) if locations is not None else 0.0
        y = float(locations["y"][i]) if locations is not None else float(i) * 25.0

        nwbfile.add_electrode(
            x             = x,
            y             = y,
            z             = 0.0,
            location      = ELECTRODE_LOCATION,
            group         = electrode_group,
        )

    all_electrodes = nwbfile.create_electrode_table_region(
        region      = list(range(len(channel_ids))),
        description = "All recorded electrodes",
    )
    n_channels = len(channel_ids)

    # ── 3e. Raw AP band ───────────────────────────────────────────────────
    if False:

        print("         - Writing raw EPhys data [...]")
 
        raw_es = ElectricalSeries(
            name             = "ElectricalSeries",
            data             = siRec,
            electrodes       = all_electrodes,
            timestamps       = timestamps,
            conversion       = 1e-6,   # µV → V  (NWB stores Volts)
            description      = "Raw Electrophysiological Data",
            comments         = f"Open-Ephys recording, {n_channels} channels",
            compression      = cfg["compression"],
        )
        nwbfile.add_acquisition(raw_es)
 
    if True:
        # ──  Sub-select channels ----------------------------------------- 
        channels_ids = siRec.get_channel_ids()
        rec_subchannels = siRec.select_channels(
                channel_ids=channels_ids[::electrode_subsampling]
        ) 
        subsampled_electrodes = nwbfile.create_electrode_table_region(
            region      = list(np.arange(len(channels_ids))[::electrode_subsampling]),
            description = "subsampled electrodes for LFP",
        )

    if True:
        # # we save the data in the memory with an **extended** chunk size
        temp_folder = os.path.join(tempfile.gettempprefix(), 'temp')
        rec_subchannels.save(format='binary', 
                            folder=temp_folder, overwrite=True,
                            chunk_duration=chunking_window,
                            n_jobs=0.8, #
                            progress_bar=True)

    # ──  LFP band ──────────────────────────────────────────────────────
    if False:

        print("         - Computing and writing LFP band [...]")

        rec = si.load(temp_folder, chunk_duration=chunking_window)

        # ── 2. Apply filter + resample pipeline on the extended chunk ─────
        rec_lfp = si.bandpass_filter(
            rec, 
            freq_min=lfp_freq_min, freq_max=lfp_freq_max,
            ignore_low_freq_error=True,
            margin_ms=margin_ms
        )
        rec_lfp = si.resample(rec_lfp, 
                            resample_rate=resample_rate)

        # ── 3. Build NWB LFP objects ───────────────────────────────────────
        lfp_es = ElectricalSeries(
            name          = "LFP",
            data          = rec_lfp.get_traces(),
            electrodes    = subsampled_electrodes,
            starting_time = 0.0,
            rate          = float(resample_rate),
            conversion    = 1e-6,   # µV → V
            description   = (
                f"LFP signal in uV "
                f"electrode subsampling: {electrode_subsampling}"
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
    if True:

        print("         - Computing and writing Multi-Unit Activity [...]")

        rec = si.load(temp_folder, chunk_duration=chunking_window)

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
            electrodes    = subsampled_electrodes,
            starting_time = 0.0,
            rate          = float(resample_rate),
            conversion    = 1e-6,   # µV → V
            description   = (
                f"MUA signal in uV "
                f"electrode subsampling: {electrode_subsampling}"
                f"MUA band ({mua_freq_min}–{mua_freq_max} Hz, "
                f"Butterworth order 5, zero-phase), "
                f"downsampled to {resample_rate} Hz. "
                f"Chunk margin: {margin_ms} ms per side, Chunking window: {chunking_window}"
            ),
        )
    
        mua_module = nwbfile.create_processing_module(
            name        = "MUA",
            description = "Local-Field Potential computed from raw electrophysiology data",
        )
        mua_module.add(mua_es)

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
 