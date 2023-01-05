import sys, os, pathlib, subprocess

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

cmd, cwd = physion.assembling.build_NWB.build_cmd(sys.argv[-1])
p = subprocess.Popen(cmd,
                     cwd=cwd,
                     shell=True)
