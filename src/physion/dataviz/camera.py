# general modules
import os
import numpy as np
import matplotlib.pylab as plt

# physion
from physion.assembling.tools import load_FaceCamera_data
from physion.utils.camera import CameraData


def loadCameraData(metadata, raw_folder):
    
    # need to have the NIdaq start:
    load_NIdaq_start(metadata, raw_folder)

    ######################
    # --- FaceCamera --- #
    ######################

    # imgfolder = os.path.join(raw_folder, 'FaceCamera-imgs')

    if os.path.isdir(imgfolder):
        times, FILES, nframes, Lx, Ly =\
                load_FaceCamera_data(imgfolder, 
                                     t0=metadata['NIdaq_Tstart'], 
                                     verbose=True)
        metadata['raw_Face_times'] = times 
        metadata['raw_Face_FILES'] = \
                [os.path.join(imgfolder, f) for f in FILES]
    else:
        metadata['raw_Face_times'] = None
        metadata['raw_Face_FILES'] = None


    ######################
    # ---  RigCamera --- #
    ######################

    imgfolder = os.path.join(raw_folder, 'RigCamera-imgs')

    if os.path.isdir(imgfolder):
        times, FILES, nframes, Lx, Ly =\
                load_FaceCamera_data(imgfolder, 
                                     t0=metadata['NIdaq_Tstart'], 
                                     verbose=True)
        metadata['raw_Rig_times'] = times 
        metadata['raw_Rig_FILES'] = \
                [os.path.join(imgfolder, f) for f in FILES]
    else:
        metadata['raw_Rig_times'] = None
        metadata['raw_Rig_FILES'] = None

    ######################
    # --- Pupil Data --- #
    ######################

    pupil_file = os.path.join(raw_folder, 'pupil.npy')

    if os.path.isfile(pupil_file):
        metadata['with_processed_pupil'] = True
        dataP = np.load(pupil_file, allow_pickle=True).item()
        for key in dataP:
            metadata['pupil_'+key] = dataP[key]
    else:
        metadata['with_processed_pupil'] = False

    ######################
    # -- Whisking Data - #
    ######################

    facemotion_file = os.path.join(raw_folder, 'facemotion.npy')

    if os.path.isfile(facemotion_file):
        metadata['with_processed_whisking'] = True
        dataP = np.load(facemotion_file, allow_pickle=True).item()
        for key in dataP:
            metadata['whisking_'+key] = dataP[key]
    else:
        metadata['with_processed_whisking'] = False

    if 'FaceCamera-1cm-in-pix' in metadata:
        # IN MILLIMETERS FROM HERE:
        metadata['pix_to_mm'] = \
                10./float(metadata['FaceCamera-1cm-in-pix']) 
    else:
        metadata['pix_to_mm'] = 1

def load_NIdaq_start(metadata, raw_folder):
   
    start = os.path.join(raw_folder, 'NIdaq.start.npy')

    if os.path.isfile(start):
        metadata['NIdaq_Tstart'] = np.load(start)[0]
    else:
        print('')
        print('---------  [!!] ------------------')
        print(' NIdaq.start.npy file not found -')
        print('  -------  [!!] ------------------')
        print('')
        print('need to deal with this [...]')


if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("raw_data_folder", type=str)

    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity", 
                        action="store_true")

    args = vars(parser.parse_args())

    metadata = np.load(os.path.join(args['raw_data_folder'],
                                    'metadata.npy'),
                       allow_pickle=True).item()

    loadCameraData(metadata, args['raw_data_folder'])

    print(metadata)
 

