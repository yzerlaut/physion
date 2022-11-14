import json, argparse, tempfile, sys, os

sys.path.append('./src')
import physion

if os.path.isdir(sys.argv[-1]):
    physion.analysis.read_NWB.scan_folder_for_NWBfiles(sys.argv[-1])
else:
    print('datafile')

