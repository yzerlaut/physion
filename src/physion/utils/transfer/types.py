"""
script to build the different ignore function to be used in shutil.copytree
    for COPYING SPECIFIC SUBSETS OF DATA
"""
import shutil

def ignore_all_behav_image_folders(Dir, f):
    return ('FaceCamera' in Dir) or ('RigCamera' in Dir)

def ignore_to_take_only_processed_imaging(Dir, files):
    print(Dir)
    print(files)
    # return [not( ('TSeries' in Dir) and\
                   # (('.npy' in f) or ('.xml' in f)) )\
                        # for f in files]
    # return [not ('TSeries' in Dir) for f in files]
    return 

def ignore_to_take_only_processed_imaging_with_vids(Dir, f):
    return (not 'TSeries' in Dir) or \
                    ((not '.npy' in f) and \
                    (not '.xml' in f) and \
                    (not '.mp4' in f) and \
                    (not '.wmv' in f))

def ignore_to_take_only_processed_imaging_with_binary(Dir, f):
    return (not 'TSeries' in Dir) or \
                        ((not '.npy' in f) and \
                        (not '.xml' in f) and \
                        (not '.bin' in f))

def ignore_to_take_only_raw_imaging(Dir, f):
    return (not 'TSeries' in Dir) or \
            ((not '.tif' in f) and \
            (not '.xml' in f))

def ignore_to_take_only_processed_behavior(Dir, f):
    return (not 'TSeries' in Dir) or\
                        ((not '.npy' in f) and\
                        (not '.xml' in f))  

def ignore_all_tiffs(Dir, f):
    return ('.tif' in f)

def ignore_to_take_only_NWBs(Dir, f):
    return (not '.nwb' in f)

def ignore_to_take_only_npy(Dir, f):
    return (not '.npy' in f)

def ignore_to_take_only_xml(Dir, f):
    return (not '.xml' in f)


TYPES = {
    'processed-Imaging':shutil.ignore_patterns('*.ome.tif', 'Reference*', 
                                               '*.avi', '*.mp4', '*.wmv',
                                               '*.env', '*.bin'),
    'processed-Imaging-wVids':shutil.ignore_patterns('*.ome.tif', 'Reference*', 
                                                     '*.env', '*.bin'),
    'processed-Imaging-wBinary':shutil.ignore_patterns('*.ome.tif', 'Reference*', 
                                                       '*.env'),
    'raw-Imaging-only':shutil.ignore_patterns('*.npy', 'suite2*', '*.env'),
    'video-Imaging-only':shutil.ignore_patterns('*.ome.tif', 
                                                'db.npy', 'ops.npy', 
                                                'suite2*', '*.env'),
    'processed-Behavior':shutil.ignore_patterns('FaceCamera-*', 
                                                'RigCamera-*',
                                                'TSerie*'),
    'nwb':shutil.ignore_patterns('*.npy', '*.env', '*.tif', '*.bin', 
                                    '*.mp4', '*.wmv', '*.avi', '*.xml'),
    'npy':shutil.ignore_patterns('*.nwb', '*.env', '*.tif', '*.bin', 
                                    '*.mp4', '*.wmv', '*.avi', '*.xml'),
    'xml':shutil.ignore_patterns('*.nwb', '*.env', '*.tif', '*.bin', 
                                    '*.mp4', '*.wmv', '*.avi', '*.npy'),
    'all':None,
}
