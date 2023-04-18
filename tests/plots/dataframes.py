import sys, os, pathlib

import numpy as np
import matplotlib.pylab as plt

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[2], 'src'))

import physion

dataframe = physion.analysis.dataframe.NWB_to_dataframe(sys.argv[-1])

settings={'CaImagingRaster':dict(fig_fraction=4, roiIndices='all'),
          'CaImaging':dict(fig_fraction=3, 
                          roiIndices=np.random.choice(np.arange(dataframe.vNrois), 
                                                     5, replace=False)),
          'Locomotion':dict(fig_fraction=1, color='#1f77b4'),
          'Pupil':dict(fig_fraction=2, color='#d62728'),
          'FaceMotion':dict(fig_fraction=1, color='tab:purple'),
          'GazeMovement':dict(fig_fraction=1, color='#ff7f0e'),
          'VisualStim':dict(fig_fraction=0, color='black')}

fig, ax = physion.dataviz.dataframe.raw.plot(dataframe, [10, 500],
                                             settings=settings, 
                                             Tbar=5)
settings={'no-show-1':dict(fig_fraction=4, roiIndices='all'),
          'CaImaging':dict(fig_fraction=3, 
                          roiIndices=np.random.choice(np.arange(dataframe.vNrois), 
                                                     5, replace=False)),
          'no-show-2':dict(fig_fraction=1, color='#1f77b4'),
          'no-show-3':dict(fig_fraction=2, color='#d62728'),
          'no-show-FaceMotion':dict(fig_fraction=1, color='tab:purple'),
          'no-show-4':dict(fig_fraction=1, color='#ff7f0e'),
          'no-show-5':dict(fig_fraction=0, color='black')}

fig, ax = physion.dataviz.dataframe.raw.plot(dataframe, [10, 500],
                                             settings=settings, 
                                             Tbar=5, ax=ax)

plt.show()

