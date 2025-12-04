"""
use as python convert datafolder
"""
import os, sys
import numpy as np
import pandas as pd
import pynwb
import datetime
from dateutil.tz import tzlocal
import pyabf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample

from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.to_nwb import add_ophys_processing_from_suite2p
from physion.utils.files import get_files_with_extension

def read_table(filename):

    dataset = pd.read_excel(filename)
                            # sheet_name='Recordings')

    return dataset

def get_pupil_diameter(Tseries_folder):
    t, pupil_diameter = None, None
    return t, pupil_diameter

def get_facemotion(Tseries_folder):
    t, pupil_diameter = None, None
    return t, pupil_diameter

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
    trace_A, trace_B = get_abf_channels(abf, ['RE1', 'RE2'])

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
    
    rec_duration = len(position) / acq_freq
    speed = resample(speed, round(new_freq*rec_duration)) #downsample the speed

    # Get downsampled timestamps
    t = np.linspace(0, rec_duration, len(speed))

    return t, speed

def get_abf_channels(abf: pyabf.ABF, channels_names: list[str]=['RE1', 'RE2']):
    
    channel_a_idx = abf.adcNames.index(channels_names[0])
    channel_b_idx = abf.adcNames.index(channels_names[1])
    
    abf.setSweep(sweepNumber=0, channel=channel_a_idx)
    ch_A = abf.sweepY

    abf.setSweep(sweepNumber=0, channel=channel_b_idx)
    ch_B = abf.sweepY

    return ch_A, ch_B

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

def convert(Tseries_folder, 
            subject_props=None):

    fn = get_files_with_extension(Tseries_folder, extension='.xml')[0]
    xml = bruker_xml_parser(fn) # metadata

    # t_imaging = xml['relativeTime']+abf.sweepEpochs.p1s[epoch_idx]

    if subject_props is not None:
        # --------------------------------------------------------------
        #    ---------  building the pynwb subject object   ----------
        # --------------------------------------------------------------
        if metadata['genotype']=='knockout':
            virus=''
            genotype='NDNF::CB1-KD'

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
                                 

    start_time = datetime.datetime(2025, 12, 4, 15, 15, 15, tzinfo=tzlocal())
    # -------------    
    #    ---------  building the pynwb NWBfile object   ----------
    # --------------------------------------------------------------
    nwbfile = pynwb.NWBFile(\
                identifier=Tseries_folder,
                session_description='imaging during spontaneous behavior',
                experiment_description='',
                experimenter='Adrianna Nozownik',
                lab='ICM Bacci lab',
                # protocol=str({k: protocol[k] for k in protocol if len(k) <66}) if metadata['protocol'] != 'None' else None,
                # institution=metadata['institution'],
                # notes=metadata['notes'],
                # virus=subject_props['virus'],
                # surgery=subject_props['surgery'],
                session_start_time=start_time,
                # subject=subject,
                # source_script=str(pathlib.Path(__file__).resolve()),
                # source_script_file_name=str(pathlib.Path(__file__).resolve()),
                file_create_date=\
                   datetime.datetime.now(datetime.UTC).replace(tzinfo=tzlocal()))

    manager = pynwb.get_manager() # we need a manager to link raw and processed data

    filename = 'temp.nwb'
    io = pynwb.NWBHDF5IO(filename,
                         mode='w', manager=manager)

    print("""     ----> Saving the NWB file: "%s" """ % filename)
    io.write(nwbfile, link_data=False)
    io.close()
    print('---> done !')


if __name__=='__main__':

    for df in os.listdir(sys.argv[-1]):
        if 'TSeries' in df:

            convert(os.path.join(sys.argv[-1], df))

