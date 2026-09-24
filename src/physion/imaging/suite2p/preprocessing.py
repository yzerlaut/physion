import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np
import h5py

from physion.utils.paths import python_path_suite2p_env
from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.default_ops import default_ops, default_settings
from physion.imaging.suite2p.presets import presets

from physion.imaging.suite2p.default_ops import default_ops


# we override some of suite2p defaults (see default_ops)
def override_suite2p_default_ops(ops, v1=False):
    ops['bruker']=True
    ops['functional_chan']= 2
    ops['align_by_chan'] = 2
    ops['batch_size'] = 500

def build_db(folder, v1=False):
    if v1:
        return {'data_path':[folder]}
    else:
        return {'data_path':[folder],
                'subfolders': [],
                'save_path0': folder,
                'fast_disk': folder,
                'input_format': 'bruker'}

#####################################################################
#  h5 input: "h5-" folders from physion.utils.compression.h5
#            (one h5 file per channel and plane, key: "data")
#####################################################################

H5_INPUT = 'suite2p-input.h5' # virtual dataset read by suite2p
H5_KEY = 'data'

def is_h5_folder(folder):
    return os.path.basename(os.path.normpath(folder)).startswith('h5-')


def is_TSeries_folder(folder):
    return os.path.basename(os.path.normpath(folder)).startswith('TSeries-')


def find_imaging_folders(folder, recursive=True):
    """
    "TSeries-" and "h5-" folders in folder (no search inside them)
        "h5-" folders are skipped if their "TSeries-" folder is still there
    """
    FOLDERS = []
    for root, subdirs, _ in os.walk(folder):
        for d in sorted(subdirs):
            if is_TSeries_folder(d) or\
                    (is_h5_folder(d) and\
                        (d.replace('h5-', 'TSeries-', 1) not in subdirs)):
                FOLDERS.append(os.path.join(root, d))
        subdirs[:] = [d for d in subdirs\
                        if not (is_TSeries_folder(d) or is_h5_folder(d))]
        if not recursive:
            break
    return sorted(FOLDERS)


def get_h5_files(folder, bruker_data):
    """
    h5 files ordered as [plane][channel], None if one is missing
    """
    planes = np.unique(\
            bruker_data[bruker_data['channels'][0]]['depth_index'])
    files = [[os.path.join(folder, '%s-plane%i.h5' %\
                                    (chan.replace(' ','-'), p))\
                    for chan in bruker_data['channels']] for p in planes]
    if np.all([os.path.isfile(f) for plane in files for f in plane]):
        return files
    else:
        return None


def build_interleaved_h5(folder, h5_files,
                         subsampling=slice(None)):
    """
    suite2p reads h5 data as a single stream of frames interleaved as:
        t0: plane0-chan0, plane0-chan1, plane1-chan0, ..., t1: ...
    we build this stream as a virtual dataset (no data copy)
        pointing to the h5 files of each channel and plane
    """
    nplanes, nchannels = len(h5_files), len(h5_files[0])
    shapes, dtypes = {}, []
    for fn in np.array(h5_files).flatten():
        with h5py.File(fn, 'r') as f:
            shapes[fn] = f[H5_KEY].shape
            dtypes.append(f[H5_KEY].dtype)
    # same number of frames for all channels & planes (interrupted acquisitions)
    nframes = min([s[0] for s in shapes.values()])
    frames = range(nframes)[subsampling] # (a VirtualSource can be sliced once)
    nsel = len(frames)

    layout = h5py.VirtualLayout(shape=(nsel*nplanes*nchannels,)+\
                                        shapes[h5_files[0][0]][1:],
                                dtype=dtypes[0])
    for p in range(nplanes):
        for c in range(nchannels):
            # relative path -> resolved from the folder of the virtual file
            source = h5py.VirtualSource(os.path.basename(h5_files[p][c]),
                                        H5_KEY, shape=shapes[h5_files[p][c]])
            layout[p*nchannels+c::nplanes*nchannels] =\
                    source[frames.start:frames.stop:frames.step]

    with h5py.File(os.path.join(folder, H5_INPUT), 'w') as f:
        f.create_virtual_dataset(H5_KEY, layout)

    print(' [ok] "%s" built: %i frames x %i planes x %i channels' %\
            (H5_INPUT, nsel, nplanes, nchannels))
    return nplanes, nchannels


def build_h5_db(folder, bruker_data, h5_files, my_settings):
    """ suite2p options to read the h5 data """

    if my_settings.get('subsampling', False):
        subsampling = slice(my_settings['subsampling_iStart'],
                            my_settings['subsampling_iStop'],
                            my_settings['subsampling_step'])
    else:
        subsampling = slice(None)

    nplanes, nchannels = build_interleaved_h5(folder, h5_files,
                                              subsampling=subsampling)

    # functional channel: "Ch2 Green" by default (see override_suite2p_default_ops)
    if (nchannels>1) and ('Ch2 Green' in bruker_data['channels']):
        functional_chan = bruker_data['channels'].index('Ch2 Green')+1
    else:
        functional_chan = 1

    return {'input_format':'h5',
            'file_list':[H5_INPUT],
            'h5py_key':H5_KEY,
            'nplanes':nplanes,
            'nchannels':nchannels,
            'functional_chan':functional_chan}


def build_suite2p_options(folder,
                          my_settings):

    xml_file = get_files_with_extension(folder, extension='.xml')[0]

    bruker_data = bruker_xml_parser(xml_file)

    if is_h5_folder(folder):
        h5_files = get_h5_files(folder, bruker_data)
        if h5_files is None:
            raise FileNotFoundError(\
                ' [!!] h5 files of "%s" missing for some channels/planes' % folder)
        h5_db = build_h5_db(folder, bruker_data, h5_files, my_settings)
    else:
        h5_db = None

    # acquisition frequency per plane - (bruker framePeriod i already per plane)
    nplanes = my_settings['nplanes']\
                        if 'nplanes' in my_settings else 1 
    acq_freq = 1./float(bruker_data['settings']['framePeriod'])/nplanes

    # hints for the size of the ROI
    um_per_pixel = float(bruker_data['settings']['micronsPerPixel']['XAxis'])
    diameter = int(my_settings['cell_diameter']/um_per_pixel) # in pixels (int 20um)
    spatial_scale = int(my_settings['cell_diameter']/6/um_per_pixel)

    if my_settings['v1']:

        settings = default_settings()
        settings['diameter'] = (diameter, diameter)
        settings['fs'] = acq_freq

        for key in my_settings:
            if key in settings:
                print(' - ', key)
                if type(settings[key]==dict):
                    for k in my_settings[key]:
                        print(10*' ', key, k, ' = ', my_settings[key][k])
                        if k in settings[key]:
                            settings[key][k] = my_settings[key][k]
                else:
                    print('         changed -> ', my_settings[key])
                    settings[key] = my_settings[key]

        np.save(os.path.join(folder, 'settings.npy'), settings)

    else:
        """ suite2p version 2.x"""

        ops = default_ops()
        override_suite2p_default_ops(ops)

        ops['fs'], ops['diameter'] = acq_freq, diameter
        ops['spatial_scale'] = spatial_scale


        # all other keys here
        for key in my_settings:
            if key in ops:
                ops[key] = my_settings[key]
    
        db = build_db(folder)
        for key in ['data_path', 'subfolders', 'save_path0',
                    'fast_disk', 'input_format']:
            ops[key] = db[key]
        if h5_db is not None:
            ops.update(h5_db)
            ops['bruker'] = False
            ops['h5py'] = [os.path.join(folder, H5_INPUT)]
            ops['align_by_chan'] = h5_db['functional_chan']
        np.save(os.path.join(folder,'ops.npy'), ops)


    # we re-build the db
    db = build_db(folder, v1=my_settings['v1'])

    if h5_db is not None:
        # h5 input (subsampling is included in the virtual dataset)
        db.update(h5_db)

    # subsampling ?
    elif my_settings['subsampling']:
        if 'Ch2 Green' in bruker_data:
            func_chan = 'Ch2 Green'
        else:
            func_chan = bruker_data['channels'][0]
            print()
            print(' took %s as the functional channel' % bruker_data['channels'][0])
            print()

        for key in ['file_list', 'tiff_list']:
            db[key] =\
                bruker_data['Ch2 Green']['tifFile'][\
                            my_settings['subsampling_iStart']:\
                            my_settings['subsampling_iStop']:\
                            my_settings['subsampling_step']]

    # save:
    np.save(os.path.join(folder,'db.npy'), db)


def run_preprocessing(args):

    if args.remove_previous and\
        (os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p'))):
       shutil.rmtree(os.path.join(args.CaImaging_folder, 'suite2p'))

    build_suite2p_options(args.CaImaging_folder, 
                          presets[args.setting_key])

    cmd = '%s -m suite2p --db "%s" --ops "%s" &' % (python_path_suite2p_env,
                                     os.path.join(args.CaImaging_folder,'db.npy'),
                                     os.path.join(args.CaImaging_folder,'ops.npy'))

    print('running "%s" \n ' % cmd)
    p = subprocess.Popen(cmd,
                         # cwd = os.path.join(pathlib.Path(__file__).resolve().parents[2], 'src'),
                         shell=True)
    

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(
    description=""" 
    Launch preprocessing of Ca-Imaging data with Suite2P
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-cf', "--CaImaging_folder", 
                        type=str, default='./')
    descr = 'Available keys :\n'
    for s in presets:
        descr += ' - %s \n' % s
    parser.add_argument('-sk', "--setting_key", 
                        type=str, default='', help=descr)
    parser.add_argument("--remove_previous", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(str(args.CaImaging_folder)) and\
            (is_TSeries_folder(args.CaImaging_folder) or\
                    is_h5_folder(args.CaImaging_folder)):
        run_preprocessing(args)
        print('--> preprocessing of "%s" done !' % args.CaImaging_folder)
    elif os.path.isdir(str(args.CaImaging_folder)):
        folders = find_imaging_folders(args.CaImaging_folder, recursive=False)
        for args.CaImaging_folder in folders:
            run_preprocessing(args)
    else:
        print('[!!] Need to provide a valid "TSeries-" or "h5-" folder [!!] ')
        








