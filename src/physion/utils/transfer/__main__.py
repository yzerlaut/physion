from .tools import TYPES

import argparse


parser=argparse.ArgumentParser()
parser.add_argument("source", type=str)
parser.add_argument("destination", type=str)
parser.add_argument("type", type=str, 
                    help="should be one of :"+str(TYPES.keys()))
args = parser.parse_args()


if args.type in TYPES:
    shutil.copytree(args.source,
                    args.destination,
                    ignore=TYPES[args.type])
else:
    print(' need to choose a key from types:')
    for k in TYPES:
        print(' - %s' % k)
