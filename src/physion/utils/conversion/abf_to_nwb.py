"""
use as python convert datafolder
"""
import os, sys
import numpy as np
import pynwb
import datetime
from dateutil.tz import tzlocal
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

def get_running_speed(Tseries_folder):
    t, speed = None, None
    return t, speed


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

