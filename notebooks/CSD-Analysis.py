# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

from elephant.current_source_density import estimate_csd
import neo
import quantities as q
from scipy.ndimage import gaussian_filter

sys.path += ['../src'] # add src code directory for physion
import physion.utils.plot_tools as pt

from physion.analysis.read_NWB import Data,\
    scan_folder_for_NWBfiles
from physion.analysis.episodes.build import EpisodeData

filename = os.path.join(os.path.expanduser('~'), 
        'DATA', 'Sally', '2026_06_09', 
        '2026_06_09-17-25-06.nwb')
        # '2026_06_09-17-45-48.nwb')

# %%
data = Data(filename)
data.build_LFP()

ep = EpisodeData(data,
                 quantities=['LFP', 'photodiode'],
                 protocol_name='flashed-stimuli')
LFP = np.transpose(ep.LFP.mean(axis=0).T-ep.LFP.mean(axis=(0,2)))
LFP = gaussian_filter(LFP, 2)

# %%
# compute the Current Source Density:
lfp = neo.AnalogSignal(LFP.T,
                       sampling_rate=1250*q.Hz,
                       units=q.microvolt)
coordinates = np.ones((ep.LFP.shape[1], 1))*q.micrometer
coordinates[:,0] *= 20*np.arange(ep.LFP.shape[1])
csd = estimate_csd(lfp,
                    method='StandardCSD',
                    coordinates=coordinates)
# smoothing
CSD = gaussian_filter(\
            np.array(csd.T), sigma=4)

# %%
fig, AX = pt.figure(axes=(1,2),
                    ax_scale=(1.5,1.5))

bounds = [-100,100]

depth_range=(10*ep.LFP.shape[1], 0)

AX[0].imshow(LFP,\
          extent=[ep.t[0], ep.t[-1], *depth_range],
          aspect='auto',
          origin='lower',
          cmap=pt.bwr,
          vmin=bounds[0], vmax=bounds[1])

pt.bar_legend(AX[0],
              bounds=bounds,
              label='LFP (uV)',
              colormap=pt.bwr)


bounds = [-.05,.05]
AX[1].imshow(CSD, 
          extent=[ep.t[0], ep.t[-1], *depth_range],
          aspect='auto',
          origin='lower',
          cmap=pt.jet,
          vmin=bounds[0], vmax=bounds[1])

pt.bar_legend(AX[1],
              bounds=bounds,
              label='CSD ($\mu$A/mm$^2$)',
              colormap=pt.jet)

for ax in AX:
    ax.set_xlim([-0.1,0.5])
    for x in [0,1]:
        ax.plot([x,x], depth_range, 'k:')
    pt.set_plot(ax, 
                xlabel='time from stim. (s)', ylabel='depth ($\mu$m)')

# %%

fig, ax = pt.figure()
ax.plot(ep.t, ep.photodiode.mean(axis=0))
ax.set_xlim([-0.5,1])
# %%
