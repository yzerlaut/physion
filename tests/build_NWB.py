import sys, os, pathlib, subprocess

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

cmd = physion.assembling.build_NWB.build_cmd(sys.argv[-1])
print(cmd)
p = subprocess.Popen(cmd,
                     shell=True)
