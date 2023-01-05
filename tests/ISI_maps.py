import sys, os, pathlib
import numpy as np

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

if 'ISImaps' in sys.argv[-1]:

    data = np.load(sys.argv[-1],
                  allow_pickle=True).item()
    trial = physion.intrinsic.RetinotopicMapping.RetinotopicMappingTrial(**data)
    trial.processTrial(isPlot=False)

    # _ = trial._getSignMap(isPlot=True)
    _ = trial._mergePatches(isPlot=True)
    physion.intrinsic.tools.plt.savefig('fig.png')
    
    physion.intrinsic.tools.plt.show()

elif os.path.isdir(sys.argv[-1]):
    maps = physion.intrinsic.tools.load_maps(sys.argv[-1])

    # physion.intrinsic.tools.plot_phase_power_maps(maps, 'up')
    # physion.intrinsic.tools.plot_retinotopic_maps(maps)

    data = physion.intrinsic.tools.build_trial_data(maps, with_params=True)

    trial = physion.intrinsic.RetinotopicMapping.RetinotopicMappingTrial(**data)
    trial.processTrial(isPlot=False)

    # _ = trial._getSignMap(isPlot=True)

    # physion.intrinsic.tools.plt.savefig('fig.png')
    physion.intrinsic.tools.plt.show()

else:
    print('pick a valid folder')

