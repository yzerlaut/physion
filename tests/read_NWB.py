import sys, os, pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

if os.path.isdir(sys.argv[-1]):
    physion.analysis.read_NWB.scan_folder_for_NWBfiles(sys.argv[-1])
else:
    data = physion.analysis.read_NWB.Data(sys.argv[-1])
    print(data.protocols)

