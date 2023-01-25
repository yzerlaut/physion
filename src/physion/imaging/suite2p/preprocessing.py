import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np

from physion.utils.paths import python_path_suite2p_env
from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.presets import ops0

defaults={'do_registration':1,
          'roidetect':True,
          'cell_diameter':20, # in um
          'tau':1.3,
          'nchannels':1,
          'functional_chan':1,
          'align_by_chan':1,
          'sparse_mode':False,
          'connected':True,
          'nonrigid':1,
          'batch_size': 500,
          'threshold_scaling':0.5,
          'mask_threshold':0.3,
          'neucoeff': 0.7}


def build_db(folder):
    db = {'data_path':[folder],
          'subfolders': [],
          'save_path0': folder,
          'fast_disk': folder,
          'input_format': 'bruker'}
    return db


def build_suite2p_options(folder,
                          settings_dict):
    
    xml_file = get_files_with_extension(folder, extension='.xml')[0]

    bruker_data = bruker_xml_parser(xml_file)
    ops = ops0.copy()

    # acquisition frequency per plane - (bruker framePeriod i already per plane)
    nplanes = settings_dict['nplanes'] if 'nplanes' in settings_dict else 1 
    ops['fs'] = 1./float(bruker_data['settings']['framePeriod'])/nplanes

    # hints for the size of the ROI
    um_per_pixel = float(bruker_data['settings']['micronsPerPixel']['XAxis'])
    ops['diameter'] = int(settings_dict['cell_diameter']/um_per_pixel) # in pixels (int 20um)
    ops['spatial_scale'] = int(settings_dict['cell_diameter']/6/um_per_pixel)

    # all other keys here
    for key in settings_dict:
        if key in ops:
            ops[key] = settings_dict[key]
    
    db = build_db(folder)
    for key in ['data_path', 'subfolders', 'save_path0',
                'fast_disk', 'input_format']:
        ops[key] = db[key]

    np.save(os.path.join(folder,'db.npy'), db)
    np.save(os.path.join(folder,'ops.npy'), ops)


def run_preprocessing(args):
    #if args.remove_previous and (os.path.isdir(os.path.join(args.CaImaging_folder, 'suite2p'))):
    #    shutil.rmtree(os.path.join(args.CaImaging_folder, 'suite2p'))
    build_suite2p_options(args.CaImaging_folder, PREPROCESSING_SETTINGS[args.setting_key])
    cmd = '%s -m suite2p --db "%s" --ops "%s" &' % (python_path_suite2p_env,
                                     os.path.join(args.CaImaging_folder,'db.npy'),
                                     os.path.join(args.CaImaging_folder,'ops.npy'))
    print('running "%s" \n ' % cmd)
    subprocess.run(cmd, shell=True)
    

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description=""" Launch preprocessing of Ca-Imaging data with Suite2P
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument('-cf', "--CaImaging_folder", type=str, default='./')
    descr = 'Available keys :\n'
    for s in PREPROCESSING_SETTINGS.keys():
        descr += ' - %s \n' % s
    parser.add_argument('-sk', "--setting_key", type=str, default='', help=descr)
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument("--silent", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(str(args.CaImaging_folder)) and ('TSeries' in str(args.CaImaging_folder)):
        run_preprocessing(args)
        # print('--> preprocessing of "%s" done !' % args.CaImaging_folder)
    elif os.path.isdir(str(args.CaImaging_folder)):
        folders = [os.path.join(args.CaImaging_folder, f) for f in os.listdir(args.CaImaging_folder) if ('TSeries' in f)]
        for args.CaImaging_folder in folders:
            run_preprocessing(args)
    else:
        print('/!\ Need to provide a valid "TSeries" folder /!\ ')
        








