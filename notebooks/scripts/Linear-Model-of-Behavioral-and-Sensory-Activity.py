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
import sys, pathlib, os, itertools
import numpy as np

# add the python path:
sys.path.append('../../src')
from physion.analysis.read_NWB import Data
from physion.analysis.dataframe import NWB_to_dataframe, extract_stim_keys
from physion.utils import plot_tools as pt


# %%
filename = os.path.join(os.path.expanduser('~'), 'CURATED' , 'NDNF-December-2022', '2022_12_14-13-27-41.nwb')
data = Data(filename)
df = NWB_to_dataframe(filename,
                      visual_stim_label='per-protocol-and-parameters-and-timepoints',
                      subsampling = 10,
                      verbose=False)

# %%
roiIndex = 0

protocols = [p for p in data.protocols if (p!='grey-10min')] # remove visual-stimulus-free protocol
fig, AX = pt.plt.subplots(5, len(protocols),
                          figsize=(7,5))
pt.plt.subplots_adjust(wspace=0.3, hspace=0.3)

STIM = extract_stim_keys(df)

for p, protocol in enumerate(protocols):
    
    varied_keys = [k for k in STIM[protocol] if k not in ['times', 'DF-key']]
    varied_values = [np.sort(np.unique(STIM[protocol][k])) for k in varied_keys]

    AX[0][p].annotate(protocol.replace('Natural-Images-4-repeats','natural-images'),
                      (0.5,1.4),
                      xycoords='axes fraction', ha='center')
    
    i=0
    for values in itertools.product(*varied_values):
       
        stim_cond = np.ones(len(STIM[protocol]['times']), dtype=bool)
        for k, v in zip(varied_keys, values):
            stim_cond = stim_cond & (STIM[protocol][k]==v)

        iTime_sorted = np.argsort(np.array(STIM[protocol]['times'])[stim_cond])
        times = np.array(STIM[protocol]['times'])[stim_cond][iTime_sorted]
        resp = []

        for t in iTime_sorted:
            stim_time_cond = df[np.array(STIM[protocol]['DF-key'])[stim_cond][t]]
            resp.append(np.mean(\
                    df['dFoF-ROI%i'%roiIndex][stim_time_cond]))
        print(times, resp)

        AX[i][p].plot(times, resp)

        AX[i][p].annotate('%s=%s' % (varied_keys, values),
                          (0,-0.1), fontsize=4,
                          rotation=90, ha='right',
                          #va='top', ha='center',
                          xycoords='axes fraction')
        
        i+=1
for ax in pt.flatten(AX):
    ax.axis('off')

pt.set_common_ylim(AX)
pt.set_common_xlim(AX)



# %%
index_cond = np.zeros(data.nwbfile.stimulus['time_start_realigned'].num_samples, dtype=int)
index_cond[2] = 1 
index_cond[37] = 1 
build_timelag_set_of_stim_specific_arrays(data, df, index_cond)

fig, ax = pt.plt.subplots(1, figsize=(5,2))
i=0
for i in np.arange(-10, 20):
    key = 'VisStim__%i'%i
    if key in df:
        ax.plot(df['time'][:100], df[key][:100]+i, color='r' if i<0 else 'k')
        i+=1
print(df)



