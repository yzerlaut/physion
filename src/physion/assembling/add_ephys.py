import numpy as np
import pynwb
# from physion.ephys.tools import filter
from spikeinterface.extractors import read_openephys
import spikeinterface.full as si
from pynwb.ecephys import (
    ElectricalSeries,
    ElectrodeGroup,
    LFP,
    SpikeEventSeries,
)

def make_lfp_recording(rec_ap: si.BaseRecording) -> si.BaseRecording:
    """Downsample AP band to LFP band (2 500 Hz → 1 250 Hz, low-pass at 300 Hz)."""
    rec_filtered = si.bandpass_filter(rec_ap, freq_min=0.5, freq_max=300.0)
    rec_lfp      = si.resample(rec_filtered, resample_rate=1250)
    return rec_lfp
 

def add_ephys(nwbfile, args,
              metadata=None):
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
            imp           = -1.0,           # unknown impedance → -1 by convention
            location      = ELECTRODE_LOCATION,
            filtering     = "None",
            group         = electrode_group,
            # label         = str(ch_id),
        )

    all_electrodes = nwbfile.create_electrode_table_region(
        region      = list(range(len(channel_ids))),
        description = "All recorded electrodes",
    )
    n_channels = len(channel_ids)

    # ── 3e. Raw AP band ───────────────────────────────────────────────────
    if False:
        print("Writing raw AP band …")
 
        chunk_frames = int(cfg["chunk_duration_sec"] * fs_ap)
        n_chunks     = int(np.ceil(n_frames / chunk_frames))
 
        # Pre-allocate a memmap-backed array to avoid loading everything into RAM
        tmp_ap = np.empty((n_frames, n_channels), dtype=np.float32)
        for i in range(n_chunks):
            start = i * chunk_frames
            end   = min(start + chunk_frames, n_frames)
            tmp_ap[start:end, :] = rec_ap.get_traces(
                start_frame=start, end_frame=end, return_scaled=True
            )   # returns µV
 
        raw_es = ElectricalSeries(
            name             = "ElectricalSeries",
            data             = tmp_ap,
            electrodes       = all_electrodes,
            starting_time    = 0.0,
            rate             = fs_ap,
            conversion       = 1e-6,   # µV → V  (NWB stores Volts)
            description      = "Raw AP band recorded by Neuropixels 1.0",
            comments         = f"Open-Ephys recording, {n_channels} channels",
            compression      = cfg["compression"],
        )
        nwbfile.add_acquisition(raw_es)
 
    # ── 3f. LFP band ──────────────────────────────────────────────────────
    if True:
        print("Computing and writing LFP band …")
        chunk_duration_sec = 1.0 # TU TUNE

        rec_lfp  = make_lfp_recording(siRec)
        fs_lfp   = rec_lfp.get_sampling_frequency()
        nf_lfp   = rec_lfp.get_num_frames()
        ck_lfp   = int(chunk_duration_sec * fs_lfp)
        n_ck_lfp = int(np.ceil(nf_lfp / ck_lfp))
 
        tmp_lfp  = np.empty((nf_lfp, n_channels), dtype=np.float32)
        for i in range(n_ck_lfp):
            start = i * ck_lfp
            end   = min(start + ck_lfp, nf_lfp)
            tmp_lfp[start:end, :] = rec_lfp.get_traces(
                start_frame=start, end_frame=end, return_scaled=True
            )
 
        lfp_electrodes = nwbfile.create_electrode_table_region(
            region      = list(range(n_channels)),
            description = "All electrodes (LFP)",
        )
 
        lfp_es = ElectricalSeries(
            name          = "LFP",
            data          = tmp_lfp,
            electrodes    = lfp_electrodes,
            starting_time = 0.0,
            rate          = float(fs_lfp),
            conversion    = 1e-6,
            description   = "LFP band (0.5–300 Hz, downsampled to 1 250 Hz)",
            compression   = 'bzip',
        )
 
        lfp_module = nwbfile.create_processing_module(
            name        = "ecephys",
            description = "Processed electrophysiology data",
        )
        lfp_module.add(LFP(electrical_series=lfp_es))
 
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
 