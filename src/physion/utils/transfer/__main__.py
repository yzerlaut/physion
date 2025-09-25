import argparse
import shutil, os, glob

from .types import TYPES

Help = ''
for t in TYPES:
    Help += ' \n - %s ' % t



parser=argparse.ArgumentParser()
parser.add_argument("source", type=str)
parser.add_argument("destination", type=str)
parser.add_argument("type", type=str, 
                    help="should be one of %s :" % Help)
args = parser.parse_args()


if args.type in TYPES:
    
    for f in [f for f in os.listdir(args.source) if not f.startswith('.')]:

        print(' - copying "%s" [...] ' % os.path.join(args.source, f))
        shutil.copytree(os.path.join(args.source, f),
                        os.path.join(args.destination, f),
                        dirs_exist_ok=True,
                        ignore=TYPES[args.type])

else:
    print(' need to choose a key from types:')
    print(Help)
