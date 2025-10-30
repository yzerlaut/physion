import os, sys, pathlib, shutil, time, datetime, tempfile, json
from PIL import Image
import numpy as np

import pynwb
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from dateutil.tz import tzlocal

from physion.behavior.locomotion import compute_speed
from physion.analysis.tools import resample, resample_signal
from physion.utils.paths import python_path
from physion.visual_stim.build import build_stim as build_visualStim

from physion.utils.camera import CameraData

from .subject import reformat_props, cleanup_keys, subject_template
from .add_ophys import add_ophys
from .realign_from_photodiode import realign_from_photodiode
from .dataset import read_spreadsheet, read_metadata
from .tools import load_FaceCamera_data,\
        build_subsampling_from_freq, StartTime_to_day_seconds

ALL_MODALITIES = ['raw_CaImaging', 'processed_CaImaging',
                  'raw_FaceCamera', 'Pupil', 'FaceMotion',
                  # 'EphysLFP', 'EphysVm',
                  'VisualStim',
                  'Locomotion'] 


def build_NWB_func(args, Subject=None):
    """
    """
    if args.verbose:
        print()
        print('=> Initializing NWB file for "%s" [...]' % args.datafolder)

    #################################################
    ####         PREPARING THE METADATA       #######
    #################################################

    metadata = read_metadata(args.datafolder)

    # add visual stimulation protocol parameters to the metadata:
    if os.path.isfile(os.path.join(args.datafolder, 'protocol.json')):
        with open(os.path.join(args.datafolder, 'protocol.json'),
                  'r', encoding='utf-8') as f:
            protocol = json.load(f)

        if protocol['Presentation']=='multiprotocol':
            # for multi-protocols we rebuild subprotocol parameters for security
            protocol['no-window'] = True
            stim = build_visualStim(protocol)
            protocol = stim.protocol

        # we add all protocol parameters to the metadata:
        for key in protocol:
            metadata[key] = protocol[key]

    # some cleanup
    if 'date' not in metadata:
        metadata['date'] = metadata['filename'][-19:-9]
        metadata['time'] = metadata['filename'][-8:]

    # replace by day and time in metadata !!
    if os.path.sep in args.datafolder:
        sep = os.path.sep
    else:
        sep = '/' # a weird behavior on Windows

    day = [int(i) for i in metadata['date'].split('_')]
    Time = [int(i) for i in metadata['time'].split('-')]
    identifier = metadata['date']+'-'+metadata['time']
    start_time = datetime.datetime(*day, *Time, tzinfo=tzlocal())

    # --------------------------------------------------------------
    #                       subject info 
    # --------------------------------------------------------------

    if Subject is not None:
        subject_props = reformat_props(Subject, 
                                       debug=args.verbose)
    else:
        subject_props = subject_template.copy()
    # some cleanup -- + calculating "age" here:
    cleanup_keys(subject_props, metadata, 
                 debug=args.verbose)

    # --------------------------------------------------------------
    #    ---------  building the pynwb subject object   ----------
    # --------------------------------------------------------------
    subject = pynwb.file.Subject(description=subject_props['description'],
                                 age=subject_props['age'],
                                 subject_id=subject_props['subject_id'],
                                 sex=subject_props['sex'],
                                 genotype=subject_props['genotype'],
                                 species=subject_props['species'],
                                 weight=subject_props['weight'],
                                 strain=subject_props['strain'],
                                 date_of_birth=\
        datetime.datetime(*subject_props['Date-of-Birth'], tzinfo=tzlocal()))
                                 

    # --------------------------------------------------------------
    #    ---------  building the pynwb NWBfile object   ----------
    # --------------------------------------------------------------
    nwbfile = pynwb.NWBFile(\
                identifier=identifier,
                session_description=str(metadata),
                experiment_description=metadata['protocol'],
                experimenter=metadata['experimenter'],
                lab=metadata['lab'],
                institution=metadata['institution'],
                notes=metadata['notes'],
                virus=subject_props['virus'],
                surgery=subject_props['surgery'],
                session_start_time=start_time,
                subject=subject,
                source_script=str(pathlib.Path(__file__).resolve()),
                source_script_file_name=str(pathlib.Path(__file__).resolve()),
                file_create_date=\
                   datetime.datetime.now(datetime.UTC).replace(tzinfo=tzlocal()))

    if not hasattr(args, 'filename') or args.filename=='':
        if args.destination_folder=='':
            args.filename = os.path.join(pathlib.Path(args.datafolder).parent,
                                         '%s.nwb' % identifier)
        else:
            args.filename = os.path.join(args.destination_folder,
                                         '%s.nwb' % identifier)

    
    manager = pynwb.get_manager() # we need a manager to link raw and processed data
    
    #################################################
    ####         IMPORTING NI-DAQ data        #######
    #################################################
    if args.verbose:
        print('=> Loading NIdaq.start timestamps data for "%s" [...]' % args.datafolder)
    try:
        NIdaq_Tstart = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
    except FileNotFoundError:
        print('   [!!] No NI-DAQ data found [!!] ')
        print('      -----> Not able to build NWB file for "%s"' % args.datafolder)
        raise BaseException

    st = datetime.datetime.fromtimestamp(NIdaq_Tstart).strftime('%H:%M:%S.%f')
    true_tstart = StartTime_to_day_seconds(st)
    
    if args.verbose:
        print('=> Loading NIdaq data for "%s" [...]' % args.datafolder)
    try:
        NIdaq_data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), 
                             allow_pickle=True).item()
    except FileNotFoundError:
        print('\n   [!!] No NI-DAQ data found [!!] \n')
        NIdaq_data = None

    # #################################################
    # ####         Locomotion                   #######
    # #################################################

    if metadata['Locomotion'] and ('Locomotion' in args.modalities):
        # compute running speed from binary NI-daq signal

        if (not args.force_recalculation_of_speed) and\
            os.path.isfile(os.path.join(args.datafolder, 'locomotion.npy')):

            if args.verbose:
                print('=> Using pre-computed running-speed for "%s" [...]' % args.datafolder)

            L = np.load(os.path.join(args.datafolder, 'locomotion.npy'), 
                         allow_pickle=True).item()
            speed, running_sampling = L['speed'], L['running_sampling']

        else:

            if args.verbose:
                print('=> Computing and storing running-speed for "%s" [...]' % args.datafolder)

            speed = compute_speed(NIdaq_data['digital'][0],
                    acq_freq=float(metadata['NIdaq-acquisition-frequency']),
                    radius_position_on_disk=float(metadata['rotating-disk']['radius-position-on-disk-cm']),
                    rotoencoder_value_per_rotation=float(metadata['rotating-disk']['roto-encoder-value-per-rotation']))
            _, speed = resample_signal(speed,
                                    original_freq=float(metadata['NIdaq-acquisition-frequency']),
                                    new_freq=args.running_sampling,
                                    pre_smoothing=2./args.running_sampling)
            running_sampling = args.running_sampling
            np.save(os.path.join(args.datafolder, 'locomotion.npy'),
                    dict(speed=speed, running_sampling=running_sampling))

        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = np.reshape(speed, (len(speed),1)),
                                   starting_time=0.,
                                   unit='cm/s',
                                   rate=running_sampling)
        nwbfile.add_acquisition(running)

    # #################################################
    # ####         Visual Stimulation           #######
    # #################################################
    
    if (metadata['VisualStim'] and ('VisualStim' in args.modalities))\
            and os.path.isfile(os.path.join(args.datafolder, 'visual-stim.npy')):

        # using Annotation TimeSeries to store the protocol parameter file
        if os.path.isfile(os.path.join(args.datafolder, 'protocol.json')):
            with open(os.path.join(args.datafolder, 'protocol.json'),
                      'r', encoding='utf-8') as f:
                protocol = json.load(f)
            nwbfile.add_trial_column(name="stim", description=str(protocol))
            nwbfile.add_trial(0., 1., stim=str(protocol))

            if protocol['Presentation']=='multiprotocol':
                i = 1
                fns = os.path.join(args.datafolder, 'subprotocols', 'Protocol-%i.json' % i)
                while os.path.isfile(fns):
                    with open(fns, 'r', encoding='utf-8') as f:
                        p = json.load(f)
                    nwbfile.add_trial(i+0., i+1., stim=str(p))
                    i+=1
                    fns = os.path.join(args.datafolder, 'subprotocols', 'Protocol-%i.json' % i)


        if NIdaq_data is not None:
            # preprocessing photodiode signal
            _, Psignal = resample_signal(NIdaq_data['analog'][0],
                                         original_freq=float(metadata['NIdaq-acquisition-frequency']),
                                         pre_smoothing=2./float(metadata['NIdaq-acquisition-frequency']),
                                         new_freq=args.photodiode_sampling)
            if args.reverse_photodiodeSignal:
               Psignal *=-1 # reversing sign on the setup
	
        VisualStim = np.load(os.path.join(args.datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()

        if 'time_duration' not in VisualStim:
            VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])

        for key in ['time_start', 'time_stop', 'time_duration']:
            metadata[key] = VisualStim[key]
            
        if (args.indices_forced is not None) and (args.times_forced is not None)\
                and (len(args.indices_forced)>0):
            print('     FORCING ALIGNEMENT IN SPECIFIC EPISODES: ', args.indices_forced)
            indices_forced=args.indices_forced
            times_forced=args.times_forced
            durations_forced=args.durations_forced

        else:
            indices_forced=(metadata['realignement_indices_forced'] if ('realignement_indices_forced' in metadata) else []),
            times_forced=(metadata['realignement_times_forced'] if ('realignement_times_forced' in metadata) else []),
            durations_forced=(metadata['realignement_durations_forced'] if ('realignement_durations_forced' in metadata) else []),

        if args.force_to_visualStimTimestamps or (NIdaq_data is None):
            if args.verbose:
                print('=> Forcing the realignement from visualStim timestamps for "%s" [...]  ' % args.datafolder)
            # we just take the original timestamps
            success = True
            for key in ['time_start', 'time_stop']:
                metadata['%s_realigned' % key] = np.array(metadata['%s' % key], dtype=float)
        else:
            # using the photodiod signal for the realignement
            if args.verbose:
                print('=> Performing realignement from photodiode for "%s" [...]  ' % args.datafolder)

            # we do the re-alignement
            success, metadata = \
                    realign_from_photodiode(Psignal, metadata,
                                    max_episode=args.max_episode,
                                    ignore_episodes=args.ignore_episodes,
                                    sampling_rate=(args.photodiode_sampling\
                                            if args.photodiode_sampling>0 else None),
                                    indices_forced=indices_forced,
                                    times_forced=times_forced,
                                    durations_forced=durations_forced,
                                    verbose=args.verbose)

        if success:
            timestamps = metadata['time_start_realigned']
            for key in ['time_start_realigned', 'time_stop_realigned']:
                VisualStimProp = pynwb.TimeSeries(name=key,
                        data = np.reshape(metadata[key][:len(timestamps)],
                                            (len(timestamps),1)),
                                  unit='seconds',
                                  timestamps=timestamps)
                nwbfile.add_stimulus(VisualStimProp)
                
            for key in VisualStim:
                None_cond = np.array([isinstance(e, type(None)) for e in VisualStim[key]]) # just checks for 'None' values
                if key in ['protocol_id', 'index']:
                    array = np.array(VisualStim[key])
                elif key in ['protocol-name']:
                    array = np.zeros(len(VisualStim['index']))
                elif (type(VisualStim[key]) in [list, np.ndarray, np.array]) and (np.sum(None_cond)>0):
                    # need to remove the None elements
                    for i in np.arange(len(VisualStim[key]))[None_cond]:
                        VisualStim[key][i] = 666 # 666 means None !!
                    array = np.array(VisualStim[key], dtype=type(np.array(VisualStim[key])[~None_cond][0]))
                else:
                    array = VisualStim[key]
                VisualStimProp = pynwb.TimeSeries(name=key,
                        data = np.reshape(array[:len(timestamps)], 
                                            (len(timestamps),1)),
                                  unit='NA',
                                  timestamps=timestamps)
                nwbfile.add_stimulus(VisualStimProp)
                
        else:
            print(' [!!] No VisualStim metadata found [!!] ')
    
        if args.verbose:
            print('=> Storing the photodiode signal for "%s" [...]' % args.datafolder)

        if NIdaq_data is not None:
            photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                          data = np.reshape(Psignal, (len(Psignal),1)),
                                          starting_time=0.,
                                          unit='[current]',
                                          rate=args.photodiode_sampling)
            nwbfile.add_acquisition(photodiode)

        
    #################################################
    ####         FaceCamera Recording         #######
    #################################################
    
    if metadata['FaceCamera']:
        
        if args.verbose:
            print('=> Storing FaceCamera acquisition for "%s" [...]' % args.datafolder)

        fcamData = CameraData('FaceCamera', folder=args.datafolder)
        FC_times = fcamData.original_times
        FC_times = check_times(FC_times, NIdaq_Tstart)
        # print(len(FC_times))

        if ('raw_FaceCamera' in args.modalities) and (len(fcamData.times)>0):
           
            imgR = fcamData.get(0)
            FC_SUBSAMPLING = build_subsampling_from_freq(args.FaceCamera_frame_sampling,
                                             1./np.mean(np.diff(fcamData.times)), 
                                            fcamData.nFrames, Nmin=3)
            def FaceCamera_frame_generator():
                for i in FC_SUBSAMPLING:
                    try:
                        im = fcamData.get(i).astype(np.uint8).reshape(imgR.shape)
                        yield im
                    except ValueError:
                        print('Pb in FaceCamera with frame #', i)
                        yield np.zeros(imgR.shape)
                        
            FC_dataI = DataChunkIterator(data=FaceCamera_frame_generator(),
                                         maxshape=(None, *imgR.shape),
                                         dtype=np.dtype(np.uint8))
            FaceCamera_frames = pynwb.image.ImageSeries(name='FaceCamera',
                                                data=FC_dataI,
                                            unit='NA',
                                        timestamps=fcamData.times[FC_SUBSAMPLING])
            nwbfile.add_acquisition(FaceCamera_frames)
            print('    [ok] raw_FaceCamera successfully added ')
                
        else:
            print('     --> no raw_FaceCamera added !! ' )

            
        #################################################
        ####         Pupil from FaceCamera        #######
        #################################################
        
        if 'Pupil' in args.modalities:

            if os.path.isfile(os.path.join(args.datafolder, 'pupil.npy')):
                
                if args.verbose:
                    print('=> Adding processed pupil data for "%s" [...]' % args.datafolder)
                    
                dataP = np.load(os.path.join(args.datafolder, 'pupil.npy'),
                                allow_pickle=True).item()

                if 'FaceCamera-1cm-in-pix' in metadata:
                    pix_to_mm = 10./float(metadata['FaceCamera-1cm-in-pix']) # IN MILLIMETERS FROM HERE
                else:
                    pix_to_mm = 1
                    
                pupil_module = nwbfile.create_processing_module(name='Pupil', 
                            description='processed quantities of Pupil dynamics,\n'+\
                    ' pupil ROI: (xmin,xmax,ymin,ymax)=(%i,%i,%i,%i)\n' % (\
                            dataP['xmin'], dataP['xmax'], dataP['ymin'], dataP['ymax'])+\
                    ' pix_to_mm=%.3f' % pix_to_mm)
                
                for key, scale in zip(['cx', 'cy', 'sx', 'sy', 'angle', 'blinking'],
                                      [pix_to_mm for i in range(4)]+[1,1]):
                    if type(dataP[key]) is np.ndarray:
                        signal = dataP[key]*scale
                        signal = resample(np.linspace(FC_times[0], FC_times[-1], len(signal)),
                                          signal, FC_times)
                        PupilProp = pynwb.TimeSeries(name=key,
                                 data = np.reshape(signal,
                                                   (len(FC_times),1)),
                                 unit='seconds',
                                 timestamps=FC_times)
                        pupil_module.add(PupilProp)

                # then add the frames subsampled
                if len(fcamData.times)>0:
                    imgP = fcamData.get(0)
                    x, y = np.meshgrid(np.arange(0,imgP.shape[0]), np.arange(0,imgP.shape[1]), indexing='ij')
                    cond = (x>=dataP['xmin']) & (x<=dataP['xmax']) & (y>=dataP['ymin']) & (y<=dataP['ymax'])

                    PUPIL_SUBSAMPLING = build_subsampling_from_freq(args.Pupil_frame_sampling,
                                                 1./np.mean(np.diff(fcamData.times)), fcamData.nFrames, Nmin=3)

                    new_shapeP = dataP['xmax']-dataP['xmin']+1, dataP['ymax']-dataP['ymin']+1
                    def Pupil_frame_generator():
                        for i in PUPIL_SUBSAMPLING:
                            try:
                                im = fcamData.get(i).astype(np.uint8)[cond].reshape(*new_shapeP)
                                yield im
                            except ValueError:
                                print('Pb in FaceCamera with frame #', i)
                                yield np.zeros(new_shapeP)
                                
                    PUC_dataI = DataChunkIterator(data=Pupil_frame_generator(),
                                                  maxshape=(None, *new_shapeP),
                                                  dtype=np.dtype(np.uint8))
                    Pupil_frames = pynwb.image.ImageSeries(name='Pupil',
                                                           data=PUC_dataI,
                                                           unit='NA',
                                                           timestamps=FC_times[PUPIL_SUBSAMPLING])
                    nwbfile.add_acquisition(Pupil_frames)

            elif os.path.isfile(os.path.join(args.datafolder, 'FaceIt','faceit.npz')):
                
                if args.verbose:
                    print('=> Adding processed pupil data for "%s" [...]' % args.datafolder)
                    
                dataP = np.load(os.path.join(args.datafolder, 'FaceIt','faceit.npz'),
                                allow_pickle=True)
                FC_timesP = FC_times[:len(dataP['pupil_dilation'])]

                if 'FaceCamera-1cm-in-pix' in metadata:
                    pix_to_mm = 10./float(metadata['FaceCamera-1cm-in-pix']) # IN MILLIMETERS FROM HERE
                else:
                    pix_to_mm = 1
                    
                pupil_module = nwbfile.create_processing_module(name='Pupil', 
                            description='processed quantities of Pupil dynamics,\n'+\
                                ' pix_to_mm=%.3f' % pix_to_mm)

                for key, key2, coef in zip(['cx', 'cy', 'sx', 'sy', 'blinking', 'area'],
                                     ['pupil_center_X', 'pupil_center_y', 'width', 'height', 
                                      'blinking_ids', 'pupil_dilation_blinking_corrected'],
                                     [pix_to_mm, pix_to_mm, pix_to_mm*2, pix_to_mm*2, 1, pix_to_mm**2]):
                    if type(dataP[key2]) is np.ndarray:
                        PupilProp = pynwb.TimeSeries(name=key,
                                 data = np.reshape(dataP[key2]*coef, 
                                                   (len(FC_timesP),1)),
                                 unit='seconds',
                                 timestamps=FC_timesP)
                        pupil_module.add(PupilProp)

            else:
                print(' [!!] No processed pupil data found',
                      'for "%s" [!!] ' % args.datafolder)

                
        #################################################
        ####      FaceMotion from FaceCamera        #######
        #################################################
    
        if 'FaceMotion' in args.modalities:
            
            if os.path.isfile(os.path.join(args.datafolder, 'facemotion.npy')):
                
                if args.verbose:
                    print('=> Adding processed facemotion data',
                          'for "%s" [...]' % args.datafolder)
                    
                dataF = np.load(os.path.join(args.datafolder, 'facemotion.npy'),
                                allow_pickle=True).item()
                FC_timesF = FC_times[:len(dataF['motion'])]

                faceMotion_module = nwbfile.create_processing_module(\
                        name='FaceMotion', 
                        description='face motion dynamics,\n'+\
                            ' facemotion ROI: (x0,dx,y0,dy)=(%i,%i,%i,%i)\n'\
                                        % (dataF['ROI'][0],dataF['ROI'][1],
                                           dataF['ROI'][2],dataF['ROI'][3]))
                signal = dataF['motion']
                signal = resample(np.linspace(FC_times[0], FC_times[-1], len(signal)),
                                    signal, FC_times)
                FaceMotionProp = pynwb.TimeSeries(name='face-motion',
                                      data = np.reshape(signal,
                                                        (len(FC_times),1)),
                                                  unit='seconds',
                                                  timestamps=FC_times)
                faceMotion_module.add(FaceMotionProp)

                if 'grooming' in dataF:
                    signal = dataF['grooming']
                    signal = resample(np.linspace(FC_times[0], FC_times[-1], len(signal)),
                                        signal, FC_times)
                    GroomingProp = pynwb.TimeSeries(name='grooming',
                                        data = np.reshape(signal,
                                                        (len(FC_times),1)),
                                                    unit='seconds',
                                                  timestamps=FC_times)
                    faceMotion_module.add(GroomingProp)

                # then add the motion frames subsampled
                if fcamData is not None:
                    
                    FACEMOTION_SUBSAMPLING=build_subsampling_from_freq(
                                        args.FaceMotion_frame_sampling,
                                        1./np.mean(np.diff(fcamData.times)),
                                        fcamData.nFrames, Nmin=3)
                    
                    imgFM = fcamData.get(0)
                    x, y = np.meshgrid(np.arange(0,imgFM.shape[0]), 
                                       np.arange(0,imgFM.shape[1]), 
                                       indexing='ij')
                    condF = (x>=dataF['ROI'][0]) &\
                            (x<=(dataF['ROI'][0]+dataF['ROI'][2])) &\
                            (y>=dataF['ROI'][1]) &\
                            (y<=(dataF['ROI'][1]+dataF['ROI'][3]))

                    new_shapeF = len(np.unique(x[condF])), len(np.unique(y[condF]))
                    
                    def FaceMotion_frame_generator():
                        for i in FACEMOTION_SUBSAMPLING:
                            i0 = np.min([i, fcamData.nFrames-2])
                            try:
                                imgFM1 = fcamData.get(i0).astype(np.uint8)[condF].reshape(*new_shapeF)
                                imgFM2 = fcamData.get(i0+1).astype(np.uint8)[condF].reshape(*new_shapeF)
                                yield imgFM2-imgFM1
                            except BaseException as be:
                                print(be)
                                print('\n Pb in FaceCamera with frame #', i)
                                yield np.zeros(new_shapeF)
            
                    FMCI_dataI = DataChunkIterator(data=FaceMotion_frame_generator(),
                                                   maxshape=(None, *new_shapeF),
                                                   dtype=np.dtype(np.uint8))
                    FaceMotion_frames = pynwb.image.ImageSeries(name='FaceMotion',
                                                                data=FMCI_dataI, unit='NA',
                                                                timestamps=FC_times[FACEMOTION_SUBSAMPLING])
                    nwbfile.add_acquisition(FaceMotion_frames)
            
            elif os.path.isfile(os.path.join(args.datafolder, 'FaceIt', 'faceit.npz')):

                if args.verbose:
                    print('=> Adding processed facemotion data',
                          'for "%s" [...]' % args.datafolder)
                    
                dataF = np.load(os.path.join(args.datafolder, 'FaceIt', 'faceit.npz'),
                                allow_pickle=True)
                FC_timesF = FC_times[:len(dataF['motion_energy'])]

                faceMotion_module = nwbfile.create_processing_module(\
                        name='FaceMotion', 
                        description='face motion dynamics,\n')
                FaceMotionProp = pynwb.TimeSeries(name='face-motion',
                                      data = np.reshape(dataF['motion_energy'],
                                                        (len(FC_timesF),1)),
                                                  unit='seconds',
                                                  timestamps=FC_timesF)
                faceMotion_module.add(FaceMotionProp)

                if not np.isnan(dataF['grooming_threshold'][0]):
                    GroomingProp = pynwb.TimeSeries(name='grooming',
                                        data = np.reshape(dataF['grooming_ids'],
                                                        (len(FC_timesF),1)),
                                                    unit='seconds',
                                                  timestamps=FC_timesF)
                    faceMotion_module.add(GroomingProp)

            else:
                print(' [!!] No processed facemotion data found for "%s" [!!] ' % args.datafolder)
                

    #################################################
    ####    Electrophysiological Recording    #######
    #################################################

    """
    iElectrophy = 1 # start on channel 1
    
    if metadata['EphysVm'] and ('EphysVm' in args.modalities):
    
        if args.verbose:
            print('=> Storing Vm signal for "%s" [...]' % args.datafolder)
            
        vm = pynwb.TimeSeries(name='Vm-Signal',
                              description='gain 1 on Multiclamp',
                              data = NIdaq_data['analog'][iElectrophy],
                              starting_time=0.,
                              unit='mV',
                              rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(vm)
        iElectrophy += 1

    if metadata['EphysLFP'] and ('EphysLFP' in args.modalities):
    
        if args.verbose:
            print('=> Storing LFP signal for "%s" [...]' % args.datafolder)
            
        lfp = pynwb.TimeSeries(name='LFP-Signal',
                               description='gain 100 on Multiclamp',
                               data = np.reshape(NIdaq_data['analog'][iElectrophy],
                                                 (len(NIdaq_data['analog'][iElectrophy]),1)),
                               starting_time=0.,
                               unit='mV',
                               rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(lfp)
    """

    #################################################
    ####         Calcium Imaging              #######
    #################################################
    # see: add_ophys.py script
    # look for 'TSeries' folder 
    TSeries = [f for f in os.listdir(args.datafolder) if 'TSeries' in f]
    if len(TSeries)==1:
        args.imaging = os.path.join(args.datafolder, TSeries[0])

        add_ophys(nwbfile, args,
                  metadata=metadata)
    else:
        print('\n[X] [!!]  Problem with the TSeries folders (either None or multiples) in "%s"  [!!] ' % args.datafolder)
    
    #################################################
    ####    add Intrinsic Imaging MAPS         ######
    #################################################
    
    
    #################################################
    ####         Writing NWB file             #######
    #################################################

    if os.path.isfile(args.filename):
        temp = str(tempfile.NamedTemporaryFile().name)+'.nwb'
        print("""
        "%s" already exists
        ---> moving the file to the temporary file directory as: "%s" [...]
        """ % (args.filename, temp))
        shutil.move(args.filename, temp)

    io = pynwb.NWBHDF5IO(args.filename, mode='w', manager=manager)
    print("""     ----> Saving the NWB file: "%s" """ % args.filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')
    
    return args.filename



def build_cmd(datafolder,
              modalities=['Locomotion', 'VisualStim'],
              force_to_visualStimTimestamps=False,
              reverse_photodiodeSignal=False,
              dest_folder=''):

    cmd = '%s -m physion.assembling.nwb %s -M ' % (python_path,
                                                   datafolder)
    cwd = os.path.join(pathlib.Path(__file__).resolve().parents[3], 'src')
    for m in modalities:
        cmd += '%s '%m
    if reverse_photodiodeSignal:
        cmd += '--reverse_photodiodeSignal '
    if force_to_visualStimTimestamps:
        cmd += '--force_to_visualStimTimestamps '
    if dest_folder!='':
        cmd += '--destination_folder %s' % dest_folder

    return cmd, cwd

def check_times(times, t0):
    # dealing with the successive substractions of NIdaq_t0
    # print(times[0], t0)
    if times[0]>1e8:
        times = times-t0
    elif times[0]<-1e8:
        times = times+t0
    # print(times[0])
    return times


if __name__=='__main__':

    import argparse, os

    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("datafolder", type=str, default='')

    parser.add_argument('-M', "--modalities", nargs='*', type=str, default=ALL_MODALITIES)

    ######## INTRODUCING A MAX EPISODE VARIABLE                     ####
    ##             IN CASE SOMETHING WENT WRONG IN THE RECORDING    ####
    parser.add_argument("--max_episode", type=int, default=-1)
    ######## AND THE POSSIBILITY TO REMOVE SPECIFIC EPISODES
    parser.add_argument("--ignore_episodes", nargs='*', type=int, default=[])
    ######## ALSO THE ABILITY TO FORCE EPISODE START AND DURATION   ####
    ##  e.g. for the protocols without the photodiode (screen off)  ####
    parser.add_argument("--indices_forced", nargs='*', type=int, default=[])
    parser.add_argument("--times_forced", nargs='*', type=float, default=[])
    parser.add_argument("--durations_forced", nargs='*', type=float, default=[])
    # or we just simply force the timestamps to the ones desired by visualStim
    parser.add_argument("--force_to_visualStimTimestamps", action="store_true")
    parser.add_argument("--reverse_photodiodeSignal", action="store_true")
    parser.add_argument("--force_recalculation_of_speed", action="store_true")

    parser.add_argument('-rs', "--running_sampling", default=50., type=float)
    parser.add_argument('-ps', "--photodiode_sampling", default=1000., type=float)
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0., type=float)
    parser.add_argument('-fcfs', "--FaceCamera_frame_sampling", default=0.001, type=float)
    parser.add_argument('-pfs', "--Pupil_frame_sampling", default=0.01, type=float)
    parser.add_argument('-sfs', "--FaceMotion_frame_sampling", default=0.005, type=float)

    parser.add_argument('-df', "--destination_folder", type=str, default='')
    parser.add_argument('-op', "--only_protocol", type=str, default='',
                        help="""
                        In recursive mode,
                        if you want to build files for a given protocol only
                        e.g.: -oe drifting-gratings 
                        """)

    parser.add_argument("--silent", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-R', "--recursive", action="store_true")
    parser.add_argument('-fi', "--files_indices", 
                        default=[0, 10000], nargs=2, type=int)

    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    args.Pupil_frame_sampling = 0
    args.FaceMotion_frame_sampling = 0
    args.FaceCamera_frame_sampling = 0

    if '.xlsx' in args.datafolder:

        filename, directory = args.datafolder, os.path.dirname(args.datafolder)
        dataset, subjects, _ = read_spreadsheet(filename)
        if args.destination_folder=='':
            args.destination_folder = os.path.join(directory, 'NWBs')
        for i in np.arange(args.files_indices[0], 
                           min([len(dataset), args.files_indices[1]])):
            print('\n \n     [%i] -- %s \n ' % (i+1, dataset['datafolder'][i]))

            # subject information:
            Subject = subjects[\
                            subjects['subject']==dataset['subject'].values[i]\
                                    ].to_dict('records')[0]

            # resetting the datafodler
            args.datafolder = dataset['datafolder'][i]
            args.filename = ''

            # building the options
            for key in ['force_to_visualStimTimestamps',
                        'reverse_photodiodeSignal']:
                setattr(args, key, True if dataset[key].values[i]=='Yes'\
                            else False)
            # building the modalities
            args.modalities = []
            for key in ALL_MODALITIES: 
                if dataset[key].values[i]=='Yes':
                    args.modalities.append(key)

            # run the build:
            build_NWB_func(args, Subject=Subject)
        
    elif args.recursive:

        i = -1
        for f, _, __ in os.walk(args.datafolder):
            timeFolder = f.split(os.path.sep)[-1]
            dateFolder = f.split(os.path.sep)[-2]
            if (len(timeFolder.split('-'))==3) and \
                    (len(dateFolder.split('_'))==3):
                i+=1
                if (i>=args.files_indices[0]) and (i<=args.files_indices[1]):
                    print()
                    print(' [ %i ]  processing "%s" [...] ' % (i,f))
                    args.datafolder = f
                    args.filename = ''
                    if args.only_protocol!='':
                        # we check that it matches the protocol
                        metadata = read_metadata(args.datafolder)
                        if args.only_protocol in metadata['protocol']:
                            build_NWB_func(args)
                        else:
                            print('')
                            print('  [!!]  ignoring:', f, ' of protocol', 
                                  metadata['protocol'])
                            print('')
                    else:
                        build_NWB_func(args)

    elif os.path.isdir(args.datafolder) and (\
                ('metadata.npy' in os.listdir(args.datafolder)) or
                       ('metadata.json' in os.listdir(args.datafolder))):
        if (args.datafolder[-1]==os.path.sep) or (args.datafolder[-1]=='/'):
            args.datafolder = args.datafolder[:-1]
        build_NWB_func(args)

    else:
        print('"%s" not a valid datafolder' % args.datafolder)
        print('                 or missing the "--recursive"/"-R" option !!')

