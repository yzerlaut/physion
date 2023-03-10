# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# general python modules for scientific analysis
import sys, pathlib, os
import numpy as np

# add the python path:
sys.path.append('../../src')
from physion.analysis.read_NWB import Data
from physion.analysis.dataframe import NWB_to_dataframe
from physion.utils import plot_tools as pt

# %%
filename = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022', '2022_12_14-13-27-41.nwb')
data = Data(filename)
df = NWB_to_dataframe(filename,
                      visual_stim_label='per-protocol',
                      subsampling = 10,
                      verbose=False)

# %%


def build_timelag_set_of_stim_specific_arrays(data, index_cond, time,
                                              pre_interval=1,
                                              post_interval=1):

    array = 0*time

    dt = np.mean(np.diff(time)) # average dt
    Nframe_pre = int(pre_interval/dt)
    Nframe_post = int(post_interval/dt)
    Nframe_stim = int(np.min([data.nwbfile.stimulus['durations'].data[i]\
            for i in flatnonzero(index_cond)])/dt)


    print(Nframe_pre, Nframe_post, Nframe_stim)
    # looping over all repeats of this index
    """
    for i in np.flatnonzero(index_cond):

        if i<data.nwbfile.stimulus['time_start_realigned'].num_samples:
            tstart = data.nwbfile.stimulus['time_start_realigned'].data[i]
            tstop = data.nwbfile.stimulus['time_stop_realigned'].data[i]

            iT0 = np.argmin((time-t0)**2)
            
            # pre
            pre_arrays, t,  = [], t0
            while (time[iT0]-tstart)<
            t_cond = (time>=tstart) & (time<tstop)
            array[t_cond] = 1
    """

    return array

build_timelag_set_of_stim_specific_arrays(data, index_cond, dataframe['time'])

fig, ax = pt.plt.subplots(1, figsize=(5,2))
t = np.arange(len(x))
ax.plot(t, 20*x, 'k-', lw=3)
cond = np.ones(len(x), dtype=bool)
i = 1
while (np.sum(cond)>0) and (i<200):
    
    cond = (x[i:]==1) & (x[:-i]==0)
    y = np.zeros(len(x)-i)
    y[cond] = 1
    ax.plot(t[:-i], y+i)
    i+=1
print(i)


