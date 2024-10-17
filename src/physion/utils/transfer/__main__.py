from .tools import TYPES

import argparse


"""

def do_not_include(Dir, f):
    return ('FaceCamera' in Dir) or ('RigCamera' in Dir)

def ignore_files(dir, files):
    return [f for f in files if (os.path.isfile(os.path.join(dir, f)) and\
            do_not_include(dir, f))]

source_folder = os.path.join(os.path.expanduser('~'), 'UNPROCESSED', '2024_01_25')
destination_folder = os.path.join(os.path.expanduser('~'), 'ASSEMBLE')

shutil.copytree(source_folder,
                os.path.join(destination_folder, 'copy'), 
                ignore=ignore_files)

"""

parser=argparse.ArgumentParser()
parser.add_argument("source", type=str)
parser.add_argument("destination", type=str)
parser.add_argument("type", type=str, 
                    help="should be one of :"+*['\n - %s ' % s for s in TYPES])
args = parser.parse_args()


shutil.copytree(args.source,
                args.destination,
                ignore=TYPES['ignore'])
