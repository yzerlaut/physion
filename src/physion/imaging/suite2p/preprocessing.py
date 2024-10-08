import sys, os, pathlib, shutil, glob, time, subprocess
import numpy as np

from physion.utils.paths import python_path_suite2p_env
from physion.utils.files import get_files_with_extension
from physion.imaging.bruker.xml_parser import bruker_xml_parser
from physion.imaging.suite2p.default_ops import default_ops
from physion.imaging.suite2p.presets import presets

from physion.imaging.suite2p.default_ops import default_ops


# we override some of suite2p defaults (see default_ops)
def override_suite2p_defaults(ops):
    ops['bruker']=True
    # no need of deconvolution yet
    ops['spikedetect']=False
    ops['functional_chan']= 2
    ops['align_by_chan'] = 2
    ops['batch_size'] = 500

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
    ops = default_ops()
    override_suite2p_defaults(ops)

    # acquisition frequency per plane - (bruker framePeriod i already per plane)
    nplanes = settings_dict['nplanes']\
                        if 'nplanes' in settings_dict else 1 
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
            ('TSeries' in str(args.CaImaging_folder)):
        run_preprocessing(args)
        print('--> preprocessing of "%s" done !' % args.CaImaging_folder)
    elif os.path.isdir(str(args.CaImaging_folder)):
        folders = [os.path.join(args.CaImaging_folder, f) for f in os.listdir(args.CaImaging_folder) if ('TSeries' in f)]
        for args.CaImaging_folder in folders:
            run_preprocessing(args)
    else:
        print('[!!] Need to provide a valid "TSeries" folder [!!] ')
        








