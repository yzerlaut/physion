import sys, os, pathlib

import numpy as np
import matplotlib.pylab as plt

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[2], 'src'))

import physion

dataframe = physion.analysis.dataframe.NWB_to_dataframe(sys.argv[-1])

settings={'CaImagingRaster':dict(fig_fraction=4,
                                 subsampling=1,
                                 roiIndices='all',
                                 normalization='per-line',
                                 subquantity='dF/F'),
          'CaImaging':dict(fig_fraction=3, subsampling=1, 
                          subquantity='dF/F', color='#2ca02c',
                          roiIndices=np.random.choice(np.arange(dataframe.vNrois), 
                                                     5, replace=False)),
          'Locomotion':dict(fig_fraction=1, subsampling=1, color='#1f77b4'),
          'Pupil':dict(fig_fraction=2, subsampling=1, color='#d62728'),
          'GazeMovement':dict(fig_fraction=1, subsampling=1, color='#ff7f0e'),
          'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey')}
          # 'VisualStim':dict(fig_fraction=.5, color='black')}

settings={'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey')}

fig, ax = physion.dataviz.dataframe.raw.plot(dataframe, [10, 500],
                                             settings=settings, 
                                             Tbar=5)

plt.show()
