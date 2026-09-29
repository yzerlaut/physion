import os
import numpy as np
import pandas as pd
from scipy import signal

from pynwb.ecephys import (
    ElectricalSeries,
    FeatureExtraction,
    SpikeEventSeries,
)
from physion.ephys.spike_sorting\
      import read_kilosort_output, fetch_good_units

def build_args_for_ephys(args, dataset, i, directory):
    args.NPX_folder = os.path.join(directory, dataset['Npx-Folder'][i])
    args.NPX_rec = dataset['Npx-Rec'][i]
    args.Location = dataset['Location'][i]
    args.LFP, args.MUA, args.Spikes =\
          dataset['LFP'][i], dataset['MUA'][i], dataset['Spikes'][i]
    args.electrode_range, args.electrode_subsampling = dataset['electrode-range'][i], dataset['electrode-subsampling'][i]
    args.bad_channels = dataset['bad-channels'][i]
    args.nStart, args.nStop = dataset['nStart'][i], dataset['nStop'][i]
    args.kilosort_folder = os.path.join(directory,
                              'kilosort4_output', 'sorter_output')
   # to update, hard-coded for now ...
    args.stream_name='Record Node 101#OneBox-100.ProbeA' 
 

def process_in_chunks(rec, process_chunk, n_out_channels,
                      resampling_factor=24,
                      chunk_duration=10., # s
                      margin=1., # s
                      n_jobs=4,
                      desc=''):
    """
    loops over the recording in chunks (with margins to avoid boundary artefacts),
        "process_chunk" filters a chunk (time, channels) sampled at full rate
        and we keep one sample every "resampling_factor"

    -> memory usage only scales with the chunk size (not the recording length)

    returns the (time, channels) array downsampled by "resampling_factor"
    """
    from concurrent.futures import ThreadPoolExecutor
    from tqdm import tqdm

    n = rec.get_num_frames()
    fs = rec.get_sampling_frequency()
    # chunks and margins as multiples of the resampling factor
    #    (so that the downsampled samples are the ones of "timestamps[::resampling_factor]")
    chunk = int(chunk_duration*fs/resampling_factor)*resampling_factor
    m = int(margin*fs/resampling_factor)*resampling_factor

    output = np.zeros((int(np.ceil(n/resampling_factor)), n_out_channels),
                      dtype=np.float32)

    def run(start):
        end = min(start+chunk, n)
        s0, s1 = max(start-m, 0), min(end+m, n)
        x = rec.get_traces(start_frame=s0, end_frame=s1).astype(np.float32)
        y = process_chunk(x, fs)
        output[start//resampling_factor:int(np.ceil(end/resampling_factor))] =\
                y[start-s0:end-s0:resampling_factor]

    starts = range(0, n, chunk)
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        for _ in tqdm(executor.map(run, starts),
                      total=len(starts), desc=desc, unit='chunk'):
            pass

    return output


def antialiasing(x, fs, resampling_factor):
    """ 
    lowpass before downsampling,
        same filter than scipy.signal.decimate (used by spikeinterface.resample):
        Chebyshev type I of order 8 at 0.8 of the new Nyquist frequency, zero-phase
    """
    sos = signal.cheby1(8, 0.05, 0.8*fs/2./resampling_factor, fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x, axis=0)


def LFP_chunk(band, resampling_factor):
    """ lowpass filter at full rate (the highpass is applied after downsampling) """
    def func(x, fs):
        sos = signal.butter(5, band[1], btype='lowpass', fs=fs, output='sos')
        return antialiasing(signal.sosfiltfilt(sos, x, axis=0), fs, resampling_factor)
    return func


def MUA_chunk(band, channel_groups, resampling_factor):
    """ bandpass, rectify, average over groups of channels """
    def func(x, fs):
        sos = signal.butter(5, band, btype='bandpass', fs=fs, output='sos')
        x = np.abs(signal.sosfiltfilt(sos, x, axis=0))
        x = np.array([x[:,g].mean(axis=1) for g in channel_groups]).T
        return antialiasing(x, fs, resampling_factor)
    return func


def add_ephys(nwbfile, args,
            metadata=None,
            LFP_BAND = [0.5, 300.0],
            MUA_BAND = [300.0, 6000.0],
            resampling_factor = 24, # int,  gives a resampled_rate = 1250,
            chunk_duration = 10., # s, processing window (memory ~ chunk size)
            n_jobs = 4):
    """
    See:
    https://pynwb.readthedocs.io/en/dev/tutorials/domain/ecephys.html
    """
    try: # optional dependency (only needed here)
        from spikeinterface.extractors import read_openephys
        import spikeinterface.full as si
    except ImportError as e:
        raise ImportError('the ephys dependencies are missing -> pip install "physion[ephys]"') from e


    #   create the device 
    device = nwbfile.create_device(
                        name="Neuropixels OneBox",
                        description="Neuropixels 2.0 probes with OneBox System\n"+\
                    "  recorded in the folder **%s**\n" % args.NPX_folder+\
                "  aligned to NIDAQ with samples: nStart=%i, nStop=%i, " % (args.nStart, args.nStop),
                        manufacturer='imec',
                    )

    #   load the open-ephys data:
    siRec = read_openephys(args.NPX_folder,
                           stream_name=args.stream_name)

    #   load the probe info
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
    # ── restrict to electrode range and remove bad channels ─────────────────

    print("         -> restricting to electrode range [...]")
    e0, e1 = [int(e) for e in args.electrode_range.split('-')]
    siRec = siRec.select_channels(siRec.get_channel_ids()[e0:e1])

    print("         -> removing bad channels [...]")
    if type(args.bad_channels) in [str, np.str_]:
        bad_channel_ids = args.bad_channels.split(',')
        siRec = siRec.remove_channels(bad_channel_ids)

    # 2)
    # ── build Electrode table ───────────────────────────────────────────────
    # 
    print("         -> building corresponding electrode table [...]")
    channel_ids = siRec.get_channel_ids()
    np.save(os.path.join(args.NPX_folder,
            'channel_ids_in_%s' % os.path.basename(args.filename).replace('nwb','py')), channel_ids)
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
        description = "Electrodes kept (in the brain + good channels)",
    )

    # 3)
    #######################################################
    # ── add Spikes ───────────────────────────────────────
    #######################################################
    if args.Spikes=='Yes':

        if os.path.isdir(args.kilosort_folder):

            # ---- read the spike sorting output from ks & phy ---- #
            # only units that have been set as "good" in manual sorting #
            data = read_kilosort_output(args.kilosort_folder)
            spike_time_indices, templates = fetch_good_units(data)

            #     ---  Spiking Module ---      #
            spiking_module = nwbfile.create_processing_module(
                name        = "Spiking",
                description = "Single Unit Module ",
            )

            print("         -> writing single-unit spike times [...]")
            #     ---  Spike times  ---        #
            for unit_id, spk_time_indices in enumerate(spike_time_indices):

                cond = (spk_time_indices>args.nStart) &\
                            (spk_time_indices<args.nStop)

                # we translate the into spike times
                spike_times = [timestamps[s-args.nStart]\
                                for s in spk_time_indices[cond]]
                # we now add to the NWB file
                nwbfile.add_unit(spike_times=spike_times,
                                electrode_group=electrode_group)

            #    ---   Spike templates   ---       #
            print("         -> writing single-unit spiking template [...]")

            # "features" should be --> time, channel, features
            #       whereas "templates" is (id, time, channel)
            spike_waveforms = FeatureExtraction(
                name="single-unit Waveforms",
                electrodes=all_electrodes,
                description=['cluster #%i' for i in range(templates.shape[0])],
                times=np.arange(templates.shape[1])/30e3,
                features=np.array([
                    [templates[:,i,k] for k in np.arange(templates.shape[2])]\
                        for i in range(templates.shape[1])])
                )
            spiking_module.add(spike_waveforms)

        else:
            print(2*"\n")
            print('   kilosort folder "%s" COULD NO BE FOUND ! ' % args.kilosort_folder)
            print(2*"\n")

    ####################################################
    ##### FROM NOW ON --> sub-selection of channels ####
    ####################################################
    if (args.LFP=='Yes') or (args.MUA=='Yes'):

        print("         -> subsampling channels for MUA and LFP [...]")

        # channel subsampling
        elecSubsampling = np.arange(len(channel_ids))[::args.electrode_subsampling]
        electrodes = nwbfile.create_electrode_table_region(
            region      = list(elecSubsampling),
            description = "Chosen electrodes in the range %s with subsampling %s" %\
                    (args.electrode_range, args.electrode_subsampling),
        )

        # resampling rate for those
        resample_rate = int(siRec.get_sampling_frequency()\
                                    /resampling_factor)

    # 4)
    #######################################################
    # ── add Multi-Unit Activity ──────────────────────────
    #######################################################
    if args.MUA=='Yes':

        print("         -> computing and writing Multi-Unit Activity [...]")

        # strategy to subsample, we do it on all channels,
        #      but we average those in between the contacts we don't keep
        channel_groups = [np.arange(ee*args.electrode_subsampling,
                                    min((ee+1)*args.electrode_subsampling,
                                        len(channel_ids)))\
                                for ee in range(len(elecSubsampling))]

        mua_traces = process_in_chunks(siRec,
                            MUA_chunk(MUA_BAND, channel_groups, resampling_factor),
                            len(channel_groups),
                            resampling_factor=resampling_factor,
                            chunk_duration=chunk_duration,
                            n_jobs=n_jobs,
                            desc='           MUA')

        # ── Build NWB MUA objects ───────────────────────────────────────
        mua_es = ElectricalSeries(
            name          = "MUA",
            data          = mua_traces,
            electrodes    = electrodes,
            timestamps    = timestamps[::resampling_factor],
            conversion    = 1e-6,   # µV → V
            description   = (
                f"MUA signal in uV "
                f"electrode channels : {args.electrode_range}"
                f"electrode subsampling: {args.electrode_subsampling}"
                f"MUA band ({MUA_BAND[0]}–{MUA_BAND[1]} Hz, "
                f"Butterworth order 5, zero-phase), rectified, averaged over the "
                f"groups of subsampled electrodes, "
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
            channel_ids = siRec.get_channel_ids()[elecSubsampling]
        ) 

        # ── 1. lowpass filter (full rate) + downsampling, in chunks
        lfp_traces = process_in_chunks(siRec,
                            LFP_chunk(LFP_BAND, resampling_factor),
                            len(elecSubsampling),
                            resampling_factor=resampling_factor,
                            chunk_duration=chunk_duration,
                            n_jobs=n_jobs,
                            desc='           LFP')

        # ── 2. highpass filter on the downsampled data (all at once, no chunk boundaries)
        print('           -> highpass filtering')
        sos = signal.butter(5, LFP_BAND[0], btype='highpass',
                            fs=siRec.get_sampling_frequency()/resampling_factor,
                            output='sos')
        lfp_traces = signal.sosfiltfilt(sos, lfp_traces, axis=0).astype(np.float32)

        # ── 3. Build NWB LFP objects ───────────────────────────────────────
        lfp_es = ElectricalSeries(
            name          = "LFP",
            data          = lfp_traces,
            electrodes    = electrodes,
            timestamps    = timestamps[::resampling_factor],
            conversion    = 1e-6,   # µV → V
            description   = (
                f"LFP signal in uV "
                f"electrode channels : {args.electrode_range}"
                f"electrode subsampling: {args.electrode_subsampling}"
                f"LFP band ({LFP_BAND[0]}–{LFP_BAND[1]} Hz, "
                f"Butterworth order 5, zero-phase: lowpass at full rate, "
                f"highpass after downsampling), "
                f"downsampled to {resample_rate} Hz. "
            ),
        )
    
        lfp_module = nwbfile.create_processing_module(
            name        = "LFP",
            description = "Local-Field Potential computed from raw electrophysiology data",
        )
        lfp_module.add(lfp_es)


if __name__=='__main__':

    print('test')
