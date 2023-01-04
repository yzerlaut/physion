import sys, os, pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

if os.path.isdir(sys.argv[-1]):
    maps = physion.intrinsic.tools.load_maps(sys.argv[-1])
    physion.intrinsic.tools.plot_phase_power_maps(maps, 'up')
    # physion.intrinsic.tools.plt.savefig('fig.png')
    physion.intrinsic.tools.plt.show()

else:
    print('pick a valid folder')

