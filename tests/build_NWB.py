import sys, os, pathlib, subprocess

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

if os.path.isfile(sys.argv[-1]) and ('.nwb' in sys.argv[-1]):
    cmd, cwd = physion.assembling.build_NWB.build_cmd(sys.argv[-1])
    p = subprocess.Popen(cmd,
                         cwd=cwd,
                         shell=True)
else:
    print(' build test data ' )

