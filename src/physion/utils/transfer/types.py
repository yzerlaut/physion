import sys, os, pathlib, shutil


def ignore_all_behav_image_folders(Dir, f):
    return ('FaceCamera' in Dir) or ('RigCamera' in Dir)

def ignore_to_take_only_processed_imaging(Dir, f):
    return (not 'TSeries' in Dir) or\
                        ((not '.npy' in f) and\
                        (not '.xml' in f))  

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
    'processed-Imaging':ignore_to_take_only_processed_imaging,
    'processed-Imaging-wVids':ignore_to_take_only_processed_imaging_with_vids,
    'processed-Imaging-wBinary':ignore_to_take_only_processed_imaging_with_binary,
    'raw-Imaging-only':ignore_to_take_only_raw_imaging,
    'processed-Behavior':ignore_to_take_only_processed_behavior,
    'stim.+behav. (processed)':ignore_all_behav_image_folders,
    'nwb':ignore_to_take_only_NWBs,
    'npy':ignore_to_take_only_npy,
    'xml':ignore_to_take_only_xml,
    'Imaging (+binary)':{},
    'all':None,
}
