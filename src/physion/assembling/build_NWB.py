import os, sys, pathlib, shutil, time, datetime, tempfile
from PIL import Image
import numpy as np

import pynwb
from hdmf.data_utils import DataChunkIterator
from hdmf.backends.hdf5.h5_utils import H5DataIO
from dateutil.tz import tzlocal

from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders
from physion.assembling.realign_from_photodiode import realign_from_photodiode
from physion.behavior.locomotion import compute_locomotion_speed
from physion.assembling.tools import build_subsampling_from_freq, load_FaceCamera_data
from physion.analysis.tools import resample_signal


ALL_MODALITIES = ['raw_CaImaging', 'processed_CaImaging',  'raw_FaceCamera',
                  'VisualStim', 'Locomotion', 'Pupil', 'FaceMotion', 'EphysLFP', 'EphysVm']


def build_NWB(args,
              Ca_Imaging_options={'Suite2P-binary-filename':'data.bin',
                                  'plane':0}):
    
    if args.verbose:
        print('Initializing NWB file for "%s" [...]' % args.datafolder)

    #################################################
    ####            BASIC metadata            #######
    #################################################
    metadata = np.load(os.path.join(args.datafolder, 'metadata.npy'),
                       allow_pickle=True).item()
    
    # replace by day and time in metadata !!
    if os.path.sep in args.datafolder:
        sep = os.path.sep
    else:
        sep = '/' # a weird behavior on Windows

    day = metadata['filename'].split('\\')[-2].split('_')
    Time = metadata['filename'].split('\\')[-1].split('-')
    identifier = metadata['filename'].split('\\')[-2]+'-'+metadata['filename'].split('\\')[-1]
    start_time = datetime.datetime(int(day[0]),int(day[1]),int(day[2]),
                int(Time[0]),int(Time[1]),int(Time[2]),tzinfo=tzlocal())

    # subject info
    if 'subject_props' in metadata and (metadata['subject_props'] is not None):
        subject_props = metadata['subject_props']
        dob = subject_props['date_of_birth'].split('_')
    else:
        subject_props = {}
        print('subject properties not in metadata ...')
        dob = ['1988', '4', '24']

    # NIdaq tstart
    if os.path.isfile(os.path.join(args.datafolder, 'NIdaq.start.npy')):
        metadata['NIdaq_Tstart'] = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]


    subject = pynwb.file.Subject(description=(subject_props['description'] if ('description' in subject_props) else 'Unknown'),
                                 sex=(subject_props['sex'] if ('sex' in subject_props) else 'Unknown'),
                                 genotype=(subject_props['genotype'] if ('genotype' in subject_props) else 'Unknown'),
                                 species=(subject_props['species'] if ('species' in subject_props) else 'Unknown'),
                                 subject_id=(subject_props['subject_id'] if ('subject_id' in subject_props) else 'Unknown'),
                                 weight=(subject_props['weight'] if ('weight' in subject_props) else 'Unknown'),
                                 date_of_birth=datetime.datetime(int(dob[0]),int(dob[1]),int(dob[2]),tzinfo=tzlocal()))
                                 
    nwbfile = pynwb.NWBFile(identifier=identifier,
                            session_description=str(metadata),
                            experiment_description=metadata['protocol'],
                            experimenter=(metadata['experimenter'] if ('experimenter' in metadata) else 'Unknown'),
                            lab=(metadata['lab'] if ('lab' in metadata) else 'Unknown'),
                            institution=(metadata['institution'] if ('institution' in metadata) else 'Unknown'),
                            notes=(metadata['notes'] if ('notes' in metadata) else 'Unknown'),
                            virus=(subject_props['virus'] if ('virus' in subject_props) else 'Unknown'),
                            surgery=(subject_props['surgery'] if ('surgery' in subject_props) else 'Unknown'),
                            session_start_time=start_time,
                            subject=subject,
                            source_script=str(pathlib.Path(__file__).resolve()),
                            source_script_file_name=str(pathlib.Path(__file__).resolve()),
                            file_create_date=datetime.datetime.utcnow().replace(tzinfo=tzlocal()))
    
    filename = os.path.join(pathlib.Path(args.datafolder).parent, '%s.nwb' % identifier)
    
    manager = pynwb.get_manager() # we need a manager to link raw and processed data
    
    #################################################
    ####         IMPORTING NI-DAQ data        #######
    #################################################
    if args.verbose:
        print('- Loading NIdaq data for "%s" [...]' % args.datafolder)
    try:
        NIdaq_data = np.load(os.path.join(args.datafolder, 'NIdaq.npy'), allow_pickle=True).item()
        NIdaq_Tstart = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
    except FileNotFoundError:
        print(' /!\ No NI-DAQ data found /!\ ')
        print('   -----> Not able to build NWB file for "%s"' % args.datafolder)
        raise BaseException

    
    true_tstart0 = np.load(os.path.join(args.datafolder, 'NIdaq.start.npy'))[0]
    st = datetime.datetime.fromtimestamp(true_tstart0).strftime('%H:%M:%S.%f')
    true_tstart = StartTime_to_day_seconds(st)
    
    # #################################################
    # ####         Locomotion                   #######
    # #################################################

    if metadata['Locomotion'] and ('Locomotion' in args.modalities):
        # compute running speed from binary NI-daq signal
        if args.verbose:
            print('- Computing and storing running-speed for "%s" [...]' % args.datafolder)

        speed = compute_locomotion_speed(NIdaq_data['digital'][0],
                                         acq_freq=float(metadata['NIdaq-acquisition-frequency']),
                                         radius_position_on_disk=float(metadata['rotating-disk']['radius-position-on-disk-cm']),
                                         rotoencoder_value_per_rotation=float(metadata['rotating-disk']['roto-encoder-value-per-rotation']))
        _, speed = resample_signal(speed,
                                   original_freq=float(metadata['NIdaq-acquisition-frequency']),
                                   new_freq=args.running_sampling,
                                   pre_smoothing=2./args.running_sampling)
        running = pynwb.TimeSeries(name='Running-Speed',
                                   data = speed,
                                   starting_time=0.,
                                   unit='cm/s',
                                   rate=args.running_sampling)
        nwbfile.add_acquisition(running)

    # #################################################
    # ####         Visual Stimulation           #######
    # #################################################
    
    if (metadata['VisualStim'] and ('VisualStim' in args.modalities)) and os.path.isfile(os.path.join(args.datafolder, 'visual-stim.npy')):

        # preprocessing photodiode signal
        _, Psignal = resample_signal(NIdaq_data['analog'][0],
                                     original_freq=float(metadata['NIdaq-acquisition-frequency']),
                                     pre_smoothing=2./float(metadata['NIdaq-acquisition-frequency']),
                                     new_freq=args.photodiode_sampling)

        VisualStim = np.load(os.path.join(args.datafolder,
                        'visual-stim.npy'), allow_pickle=True).item()
        # using the photodiod signal for the realignement
        if args.verbose:
            print('=> Performing realignement from photodiode for "%s" [...]  ' % args.datafolder)
        if 'time_duration' not in VisualStim:
            VisualStim['time_duration'] = np.array(VisualStim['time_stop'])-np.array(VisualStim['time_start'])
        for key in ['time_start', 'time_stop', 'time_duration']:
            metadata[key] = VisualStim[key]
            
        success, metadata = realign_from_photodiode(Psignal, metadata,
                                                    sampling_rate=(args.photodiode_sampling if args.photodiode_sampling>0 else None),
                                                    indices_forced=(metadata['realignement_indices_forced'] if ('realignement_indices_forced' in metadata) else []),
                                                    times_forced=(metadata['realignement_times_forced'] if ('realignement_times_forced' in metadata) else []),
                                                    durations_forced=(metadata['realignement_durations_forced'] if ('realignement_durations_forced' in metadata) else []),
                                                    verbose=args.verbose)

        if success:
            timestamps = metadata['time_start_realigned']
            for key in ['time_start_realigned', 'time_stop_realigned']:
                VisualStimProp = pynwb.TimeSeries(name=key,
                                                  data = metadata[key],
                                                  unit='seconds',
                                                  timestamps=timestamps)
                nwbfile.add_stimulus(VisualStimProp)
                
            for key in VisualStim:
                None_cond = np.array([isinstance(e, type(None)) for e in VisualStim[key]]) # just checks for 'None' values
                if key in ['protocol_id', 'index']:
                    array = np.array(VisualStim[key])
                elif key in ['protocol-name']:
                    array = np.array([0])
                elif (type(VisualStim[key]) in [list, np.ndarray, np.array]) and (np.sum(None_cond)>0):
                    # need to remove the None elements
                    for i in np.arange(len(VisualStim[key]))[None_cond]:
                        VisualStim[key][i] = 666 # 666 means None !!
                    array = np.array(VisualStim[key], dtype=type(np.array(VisualStim[key])[~None_cond][0]))
                else:
                    array = VisualStim[key]
                VisualStimProp = pynwb.TimeSeries(name=key,
                                                  data = array,
                                                  unit='NA',
                                                  timestamps=timestamps)
                nwbfile.add_stimulus(VisualStimProp)
                
        else:
            print(' /!\ No VisualStim metadata found /!\ ')
        #     # print('   -----> Not able to build NWB file for "%s" ' % args.datafolder)
        #     # TEMPORARY FOR TROUBLESHOOTING !!
        #     metadata['time_start_realigned'] = metadata['time_start']
        #     metadata['time_stop_realigned'] = metadata['time_stop']
        #     print(' /!\ Realignement unsuccessful /!\ ')
        #     print('       --> using the default time_start / time_stop values ')
    
        if args.verbose:
            print('=> Storing the photodiode signal for "%s" [...]' % args.datafolder)

        photodiode = pynwb.TimeSeries(name='Photodiode-Signal',
                                      data = Psignal,
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

            
        try:

            FC_FILES, FC_times, FCS_data = None, None, None
            if os.path.isdir(os.path.join(args.datafolder, 'FaceCamera-imgs')):
                FC_times, FC_FILES, _, _, _ = load_FaceCamera_data(os.path.join(args.datafolder, 'FaceCamera-imgs'),
                                                                   t0=NIdaq_Tstart,
                                                                   verbose=True)
            else:
                FCS_data = np.load(os.path.join(args.datafolder, 'FaceCamera-summary.npy'), allow_pickle=True).item()
                FC_times = FCS_data['times']

            if ('raw_FaceCamera' in args.modalities) and (FC_FILES is not None):
                
                    imgR = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[0]))
                    FC_SUBSAMPLING = build_subsampling_from_freq(args.FaceCamera_frame_sampling,
                                                                 1./np.mean(np.diff(FC_times)), len(FC_FILES), Nmin=3)
                    def FaceCamera_frame_generator():
                        for i in FC_SUBSAMPLING:
                            try:
                                im = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i])).astype(np.uint8).reshape(imgR.shape)
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
                                                                timestamps=FC_times[FC_SUBSAMPLING])
                    nwbfile.add_acquisition(FaceCamera_frames)
                    
            elif ('raw_FaceCamera' in args.modalities) and (FCS_data is not None):

                    imgR = FCS_data['sample_frames'][0]
                    def FaceCamera_frame_generator():
                        for i in range(len(FCS_data['sample_frames'])):
                            try:
                                yield FCS_data['sample_frames'][i].astype(np.uint8).reshape(imgR.shape)
                            except ValueError:
                                print('Pb in FaceCamera with frame #', i)
                                yield np.zeros(imgR.shape)[shape_cond]
                                
                    FC_dataI = DataChunkIterator(data=FaceCamera_frame_generator(),
                                                 maxshape=(None, *imgR.shape),
                                                 dtype=np.dtype(np.uint8))
                    FaceCamera_frames = pynwb.image.ImageSeries(name='FaceCamera',
                                                                data=FC_dataI,
                                                                unit='NA',
                                                                timestamps=FC_times[np.linspace(0, len(FC_times)-1, len(FCS_data['sample_frames']), dtype=int)])
                                                                # timestamps=FC_times[FCS_data['sample_frames_index']]) # REPLACE THE ABOVE LINE AFTER SEPT 16th !!!
                    nwbfile.add_acquisition(FaceCamera_frames)
                    
            else:
                print(' --> no raw_FaceCamera added !! ' )

        except BaseException as be:
            print(be)
            print(' /!\ Problems with FaceCamera data for "%s" /!\ ' % args.datafolder)
            

        #################################################
        ####         Pupil from FaceCamera        #######
        #################################################
        
        if 'Pupil' in args.modalities:

            # add_pupil_data(nwbfile, FC_FILES, args)
            
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
                                                                ' pupil ROI: (xmin,xmax,ymin,ymax)=(%i,%i,%i,%i)\n' % (dataP['xmin'], dataP['xmax'], dataP['ymin'], dataP['ymax'])+\
                                                                ' pix_to_mm=%.3f' % pix_to_mm)
                
                for key, scale in zip(['cx', 'cy', 'sx', 'sy', 'angle', 'blinking'], [pix_to_mm for i in range(4)]+[1,1]):
                    if type(dataP[key]) is np.ndarray:
                        PupilProp = pynwb.TimeSeries(name=key,
                                                     data = dataP[key]*scale,
                                                     unit='seconds',
                                                     timestamps=FC_times[dataP['frame']])
                        pupil_module.add(PupilProp)

                # then add the frames subsampled
                if FC_FILES is not None:
                    imgP = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[0]))
                    x, y = np.meshgrid(np.arange(0,imgP.shape[0]), np.arange(0,imgP.shape[1]), indexing='ij')
                    cond = (x>=dataP['xmin']) & (x<=dataP['xmax']) & (y>=dataP['ymin']) & (y<=dataP['ymax'])

                    PUPIL_SUBSAMPLING = build_subsampling_from_freq(args.Pupil_frame_sampling,
                                                                    1./np.mean(np.diff(FC_times)), len(FC_FILES), Nmin=3)

                    new_shapeP = dataP['xmax']-dataP['xmin']+1, dataP['ymax']-dataP['ymin']+1
                    def Pupil_frame_generator():
                        for i in PUPIL_SUBSAMPLING:
                            try:
                                im = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i])).astype(np.uint8)[cond].reshape(*new_shapeP)
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
                        
            else:
                print(' /!\ No processed pupil data found for "%s" /!\ ' % args.datafolder)

                
    
        #################################################
        ####      FaceMotion from FaceCamera        #######
        #################################################
    
        if 'FaceMotion' in args.modalities:
            
            if os.path.isfile(os.path.join(args.datafolder, 'facemotion.npy')):
                
                if args.verbose:
                    print('=> Adding processed facemotion data for "%s" [...]' % args.datafolder)
                    
                dataF = np.load(os.path.join(args.datafolder, 'facemotion.npy'),
                                allow_pickle=True).item()

                faceMotion_module = nwbfile.create_processing_module(name='FaceMotion', 
                                                                     description='face motion dynamics,\n'+\
                                                                     ' facemotion ROI: (x0,dx,y0,dy)=(%i,%i,%i,%i)\n' % (dataF['ROI'][0],dataF['ROI'][1],
                                                                                                                         dataF['ROI'][2],dataF['ROI'][3]))
                FaceMotionProp = pynwb.TimeSeries(name='face-motion',
                                                  data = dataF['motion'],
                                                  unit='seconds',
                                                  timestamps=FC_times[dataF['frame']])
                faceMotion_module.add(FaceMotionProp)

                if 'grooming' in dataF:
                    GroomingProp = pynwb.TimeSeries(name='grooming',
                                                    data = dataF['grooming'],
                                                    unit='seconds',
                                                    timestamps=FC_times[dataF['frame']])
                    faceMotion_module.add(GroomingProp)

                # then add the motion frames subsampled
                if FC_FILES is not None:
                    
                    FACEMOTION_SUBSAMPLING = build_subsampling_from_freq(args.FaceMotion_frame_sampling,
                                                                         1./np.mean(np.diff(FC_times)), len(FC_FILES), Nmin=3)
                    
                    imgFM = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[0]))
                    x, y = np.meshgrid(np.arange(0,imgFM.shape[0]), np.arange(0,imgFM.shape[1]), indexing='ij')
                    condF = (x>=dataF['ROI'][0]) & (x<=(dataF['ROI'][0]+dataF['ROI'][2])) &\
                        (y>=dataF['ROI'][1]) & (y<=(dataF['ROI'][1]+dataF['ROI'][3]))

                    new_shapeF = len(np.unique(x[condF])), len(np.unique(y[condF]))
                    
                    def FaceMotion_frame_generator():
                        for i in FACEMOTION_SUBSAMPLING:
                            i0 = np.min([i, len(FC_FILES)-2])
                            try:
                                imgFM1 = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i0])).astype(np.uint8)[condF].reshape(*new_shapeF)
                                imgFM2 = np.load(os.path.join(args.datafolder, 'FaceCamera-imgs', FC_FILES[i0+1])).astype(np.uint8)[condF].reshape(*new_shapeF)
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
                        
            else:
                print(' /!\ No processed facemotion data found for "%s" /!\ ' % args.datafolder)
                

    #################################################
    ####    Electrophysiological Recording    #######
    #################################################

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
                               data = NIdaq_data['analog'][iElectrophy],
                               starting_time=0.,
                               unit='mV',
                               rate=float(metadata['NIdaq-acquisition-frequency']))
        nwbfile.add_acquisition(lfp)
        
    #################################################
    ####         Calcium Imaging              #######
    #################################################
    # see: add_ophys.py script
    Ca_data = None

    #################################################
    ####         Writing NWB file             #######
    #################################################

    if os.path.isfile(filename):
        temp = str(tempfile.NamedTemporaryFile().name)+'.nwb'
        print("""
        "%s" already exists
        ---> moving the file to the temporary file directory as: "%s" [...]
        """ % (filename, temp))
        shutil.move(filename, temp)
        print('---> done !')

    io = pynwb.NWBHDF5IO(filename, mode='w', manager=manager)
    print("""
    ---> Creating the NWB file: "%s"
    """ % filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')
    
    if Ca_data is not None:
        Ca_data.close() # can be closed only after having written

    return filename
    
if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', "--compression", type=int, default=0,
                        help='compression level, from 0 (no compression) to 9 (large compression, SLOW)')
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-rf', "--root_datafolder", type=str, default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-m', "--modalities", nargs='*', type=str, default=ALL_MODALITIES)
    parser.add_argument('-d', "--day", type=str, default=datetime.datetime.today().strftime('%Y_%m_%d'))
    parser.add_argument('-t', "--time", type=str, default='')
    parser.add_argument('-e', "--export", type=str, default='FROM_VISUALSTIM_SETUP',
                        help='export option [FULL / LIGHTWEIGHT / FROM_VISUALSTIM_SETUP]')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-rs', "--running_sampling", default=50., type=float)
    parser.add_argument('-ps', "--photodiode_sampling", default=1000., type=float)
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0., type=float)
    parser.add_argument('-fcfs', "--FaceCamera_frame_sampling", default=0.001, type=float)
    parser.add_argument('-pfs', "--Pupil_frame_sampling", default=0.01, type=float)
    parser.add_argument('-sfs', "--FaceMotion_frame_sampling", default=0.005, type=float)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument('-lw', "--lightweight", action="store_true")
    parser.add_argument('-fvs', "--from_visualstim_setup", action="store_true")
    parser.add_argument('-ndo', "--nidaq_only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--standard", action="store_true")
    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    # some pre-defined settings here
    if args.export=='LIGHTWEIGHT' or args.lightweight:
        args.export='LIGHTWEIGHT'
        # 0 values for all (means 3 frame, start-middle-end)
        args.Pupil_frame_sampling = 0
        args.FaceMotion_frame_sampling = 0
        args.FaceCamera_frame_sampling = 0
        args.CaImaging_frame_sampling = 0
    if args.export=='FULL' or args.full:
        args.export='FULL'
        # push all to very high values
        args.CaImaging_frame_sampling = 1e5
        args.Pupil_frame_sampling = 1e5
        args.FaceMotion_frame_sampling = 1e5
        args.FaceCamera_frame_sampling = 0.5 # no need to have it too high
    if args.nidaq_only:
        args.export='NIDAQ'
        args.modalities = ['VisualStim', 'Electrophy']        

    if args.time!='':
        args.datafolder = os.path.join(args.root_datafolder, args.day, args.time)

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            if (args.datafolder[-1]==os.path.sep) or (args.datafolder[-1]=='/'):
                args.datafolder = args.datafolder[:-1]
            build_NWB(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
    elif args.root_datafolder!='':
        FOLDERS = [l for l in os.listdir(args.root_datafolder) if len(l)==8]
        for f in FOLDERS:
            args.datafolder = os.path.join(args.root_datafolder, f)
            try:
                build_NWB(args)
            except BaseException as e:
                print(e)
