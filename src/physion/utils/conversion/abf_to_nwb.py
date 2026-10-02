"""
use as python convert datafolder
"""
import os, sys, pathlib
import numpy as np
import pandas as pd
import pandas as pd
import pynwb
import datetime
from dateutil.tz import tzlocal
import pyabf, json
from scipy.ndimage import gaussian_filter1d

from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.to_nwb import add_ophys_processing_from_suite2p
from physion.utils.files import get_files_with_extension
from physion.analysis.tools import resample_signal
from physion.analysis.tools import resample_signal

def read_table(filename):

    dataset = pd.read_excel(filename,
                            sheet_name='Recordings')
    
    return dataset

def get_face_metrics(Tseries_folder):

    fn = get_files_with_extension(os.path.join(Tseries_folder, 'FaceIt'), extension='.npz')[0]
    faceitOutput = np.load(fn, allow_pickle=True)

    output = {}
    output['cx'] = faceitOutput['pupil_center_X']
    output['cy'] = faceitOutput['pupil_center_y']

    # WE TRICK THE PUPIL COORDINATES TO HAVE SOMETHING CORRECTED:
    output['sx'] = faceitOutput['pupil_dilation_blinking_corrected']/2./np.pi # faceitOutput['width']
    output['sy'] = np.ones(len(output['cx'])) # faceitOutput['height']
    output['blinking'] = faceitOutput['blinking_ids']

    output['grooming'] = faceitOutput['grooming_ids']
    output['face-motion'] = faceitOutput['motion_energy_without_grooming']

    return output

def get_running_speed(Tseries_folder, 
                      bin_conversion_th=1.5,
                      perimeter_cm=29,
                      cpr=1000,
                      new_freq=30,
                      position_smoothing=10e-3):

    fn = get_files_with_extension(Tseries_folder, extension='.abf')[0]
    abf = pyabf.ABF(fn)

    # Get acquisition frequency
    acq_freq = abf.sampleRate

    # Get channels
    trace_A, trace_B, t = get_abf_channels(abf, ['RE1', 'RE2'])

    # Concert to binary
    bin_A = np.where(trace_A >= bin_conversion_th, 1, 0)
    bin_B = np.where(trace_B >= bin_conversion_th, 1, 0)

    # Compute position
    position = compute_position_from_binary_signals(bin_A, bin_B, forward='counterclockwise')*perimeter_cm/cpr/4. #counts per revolution is cpr * 4
    if position_smoothing>0:
        position = gaussian_filter1d(position, int(position_smoothing*acq_freq), mode='nearest')

    # Compute position
    speed_vector = np.gradient(position, 1./acq_freq)
    speed = np.abs(speed_vector) #get the norm of the speed vector
    
    # Downsample the speed
    t, speed = resample_signal(speed, 
                               t_sample=t,
                               new_freq=new_freq)
    
    # Get start time of calcium imaging recording relative to rotary encoder recording
    abf.setSweep(sweepNumber=0, channel=0)
    epoch_idx = abf.sweepEpochs.types.index('Pulse')
    dt_start = abf.sweepEpochs.p1s[epoch_idx] / abf.sampleRate

    return t, speed, dt_start

def get_abf_channels(abf: pyabf.ABF, channels_names: list[str]=['RE1', 'RE2']):
    
    channel_a_idx = abf.adcNames.index(channels_names[0])
    channel_b_idx = abf.adcNames.index(channels_names[1])
    
    abf.setSweep(sweepNumber=0, channel=channel_a_idx)
    ch_A = abf.sweepY

    abf.setSweep(sweepNumber=0, channel=channel_b_idx)
    ch_B = abf.sweepY

    t = abf.sweepX

    return ch_A, ch_B, t

def compute_position_from_binary_signals(A, B, forward='counterclockwise'):
    '''
    Takes traces A and B and converts it to a trace that has the same number of
    points but with positions points.

    Algorithm based on the schematic of cases shown in the doc
    ---------------
    Input:
        A, B - traces to convert
   
    Output:
        Positions through time

    '''

    Delta_position = np.zeros(len(A)-1, dtype=float) # N-1 elements

    ################################
    ## positive_increment_cond #####
    ################################
    # The A signal lead the B signal (counterclockwise)
    # ... => 11 => 01 => 00 => 10 => 11 => ...
    PIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[PIC] = 1

    ################################
    ## negative_increment_cond #####
    ################################
    # The B signal lead the A signal (clockwise)
    # ... => 11 => 10 => 00 => 01 => 11 => ...
    NIC = ( (A[:-1]==1) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==0) ) | \
        ( (A[:-1]==1) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==0) ) | \
        ( (A[:-1]==0) & (B[:-1]==0) & (A[1:]==0) & (B[1:]==1) ) | \
        ( (A[:-1]==0) & (B[:-1]==1) & (A[1:]==1) & (B[1:]==1) )
    Delta_position[NIC] = -1

    if forward=='clockwise':
        Delta_position = -Delta_position

    return np.cumsum(np.concatenate([[0], Delta_position]))

def convert(Tseries_folder, day, 
            subject, genotype, virus):

    # load metadata:
    fn = get_files_with_extension(Tseries_folder, extension='.txt')[0]
    with open(fn, 'r') as f:
        metadata = json.load(f)

    # load Prairie file
    fn = get_files_with_extension(Tseries_folder, extension='.xml')[0]
    xml = bruker_xml_parser(fn)
    time = [int(t[:2]) for t in xml['StartTime'].split(':')]
    day = [int(d) for d in day.split('_')]

    nwb_filename = '%i_%.2d_%.2d-%i-%i-%i.nwb' % (*day, *time)

    # --------------------------------------------------------------
    #    ---------  building the pynwb subject object   ----------
    # --------------------------------------------------------------

    subject = pynwb.file.Subject(description=subject,
                                    age='P90',
                                    subject_id=metadata['Mouse_Code'],
                                    sex=metadata['Sex'],
                                    genotype=genotype,
                                    species='mus musculus',
                                    strain='C57BL6')
                                 

    start_time = datetime.datetime(day[0], day[1], day[2], 
                                   time[0], time[1], time[2], 
                                   tzinfo=tzlocal())
    # -------------    
    #    ---------  building the pynwb NWBfile object   ----------
    # --------------------------------------------------------------
    nwbfile = pynwb.NWBFile(\
                identifier=Tseries_folder,
                session_description=str(metadata),
                experiment_description='imaging during spontaneous behavior',
                experimenter='Adrianna Nozownik',
                lab='Bacci lab',
                # protocol='spontaneous-activity',
                institution='Paris Brain Institute',
                virus=virus,
                surgery='Viral-Injection+Headplate-Implantation',
                session_start_time=start_time,
                subject=subject,
                source_script=str(pathlib.Path(__file__).resolve()),
                source_script_file_name=str(pathlib.Path(__file__).resolve()),
                file_create_date=\
                   datetime.datetime.now(datetime.UTC).replace(tzinfo=tzlocal()))

    manager = pynwb.get_manager() # we need a manager to link raw and processed data

    # #################################################
    # ####         Locomotion                   #######
    # #################################################

    t_speed, speed, dt_start = get_running_speed(Tseries_folder)

    running = pynwb.TimeSeries(name='Running-Speed',
                               data = np.reshape(speed, (len(speed),1)),
                               timestamps=t_speed,
                               unit='cm/s')
    nwbfile.add_acquisition(running)


    #################################################
    #### Pupil and Facemotion from FaceCamera #######
    #################################################

    fn = get_files_with_extension(Tseries_folder, extension='.xml')[0]
    xml = bruker_xml_parser(fn) # metadata

    t_imaging = xml['Green']['relativeTime'] + dt_start

    faceMetrics = get_face_metrics(Tseries_folder)
    t_facedata = np.linspace(t_imaging[0], t_imaging[-1], len(faceMetrics['cx']))

    # Pupil
    pupil_module = nwbfile.create_processing_module(name='Pupil',
                                                    description='')
    
    for key in ['cx', 'cy', 'sx', 'sy', 'blinking']:
        print(key)
        PupilProp = pynwb.TimeSeries(name=key,
                    data = np.reshape(faceMetrics[key],
                                    (len(t_facedata),1)),
                    unit='seconds',
                    timestamps=t_facedata)
        pupil_module.add(PupilProp)

    # FaceMotion
    faceMotion_module = nwbfile.create_processing_module(name='FaceMotion', 
                                                         description='')
    
    for key in ['face-motion', 'grooming']:
        faceMotionProp = pynwb.TimeSeries(name=key,
                    data = np.reshape(faceMetrics[key],
                                    (len(t_facedata),1)),
                    unit='seconds',
                    timestamps=t_facedata)
        faceMotion_module.add(faceMotionProp)


    #################################################
    ####         Adding Imaging               #######
    #################################################
    functional_chan = 'Ch1'
    laser_key = 'Excitation 1'
    Depth = float(xml['settings']['positionCurrent']['ZAxis'])

    device = pynwb.ophys.Device(\
        'Imaging device with settings %s' %\
         str(xml['settings']).replace(': ','= '))
    nwbfile.add_device(device)
    optical_channel = pynwb.ophys.OpticalChannel(\
            'excitation_channel 1',
             laser_key,
             float(xml['settings']['laserWavelength'][laser_key]))

    imaging_plane = nwbfile.create_imaging_plane(\
            'my_imgpln', optical_channel,
                description='Depth=%.1f[um]' % Depth,
                device=device,
                excitation_lambda=float(xml['settings']['laserWavelength'][laser_key]),
                imaging_rate=1./float(xml['settings']['framePeriod']),
                indicator='GCamp',
                location='V1', # ADD METADATA HERE
                # reference_frame='A frame to refer to',
                grid_spacing=(\
                        float(xml['settings']['micronsPerPixel']['YAxis']),
                        float(xml['settings']['micronsPerPixel']['XAxis'])))

    image_series = pynwb.ophys.TwoPhotonSeries(\
           name='CaImaging-TimeSeries',
           dimension=[2], 
           data=np.ones((2,2,2)),
           imaging_plane=imaging_plane, 
           unit='s', 
           timestamps=1.*np.arange(2), # ADD UPDATE OF starting_time
           comments='raw-data-folder=%s' % Tseries_folder.replace('/', '**')) # TEMPORARY
    
    nwbfile.add_acquisition(image_series)

    if os.path.isdir(os.path.join(Tseries_folder, 'suite2p')):
        print('=> Adding the suite2p processing for "%s" [...]' % Tseries_folder)
        add_ophys_processing_from_suite2p(os.path.join(Tseries_folder, 'suite2p'),
                                          nwbfile, xml,
                                          TwoP_trigger_delay=dt_start,
                                          device=device,
                                          optical_channel=optical_channel,
                                          imaging_plane=imaging_plane,
                                          image_series=image_series) 
    else:
        print('\n [!!]  no "suite2p" folder found in "%s"  [!!] ' % Tseries_folder)

    #################################################
    ####         Writing NWB file             #######
    #################################################

    filename = os.path.join(os.path.dirname(Tseries_folder), 
                            '..', '..',
                            'NWBs', nwb_filename)
    io = pynwb.NWBHDF5IO(filename,
                         mode='w', manager=manager)

    print("""     ----> Saving the NWB file: "%s" """ % filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')


if __name__=='__main__':

    if '.xlsx' in sys.argv[-1]:

        dataset = read_table(sys.argv[-1])    

        for f, subject, virus, genotype in zip(\
                            dataset['filepath'],
                            dataset['subject'],
                            dataset['virus'],
                            dataset['genotype']):

            day, tseries = f.split('\\')[-2:]
            tS = os.path.join(os.path.dirname(sys.argv[-1]), 
                              'processed', day, tseries)

            convert(tS, day, subject, virus, genotype)

    else:

        print("""

        [!!] need to provide a DataTable.xlsx file as argument [!!]

              """)

