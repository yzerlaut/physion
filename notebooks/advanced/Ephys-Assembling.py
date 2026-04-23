
# %% [markdown]
#
# # Assemble Neuropixels data
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```
# ## 1) 
# cd physion/src
# Run:
# ```
# python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA\2026_02_13
# ```
# this will create a file: `%USERPROFILE%\DATA\DataTable0.xlsx`  
#      move it to  ~/DATA/2026_02_13/DataTable0.xlsx
#
# Then fill its neuropixels folder (`Npx-Folder`, for example 2026-03-19_16-13-00) and recordings information (`Npx-Rec`for example node0/exp1/rec1).    
#
#       N.B. you can use the code below to guide filling the recordings info

# %%
import sys, time
sys.path += [os.path.expanduser('~/physion/src'), '../../src']
import json
import numpy as np
import pandas as pd

from open_ephys.analysis import Session as OpenEphysSession
from physion.assembling.dataset import read_spreadsheet

################################################
###   code to do the alignement !!      ########
################################################
from physion.ephys.alignement import load_nidaq_synch_signal,\
    load_OpenEphys, sampling_match

import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_04_10').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
PROBE_NAME = 'ProbeA'
EXP = 1 # 
NODE=0

# %% [markdown]
#
# ## Load Table data

# %%

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable0.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
DF = pd.DataFrame(columns=['time', 'Npx-Rec', 'daq-nEpisodes', 'ephys-nEpisodes', 'i0', 'i1', 'nStart', 'nStop'])
DF['time'] = datatable['time']
DF['Npx-Folder'] = datatable['Npx-Folder']

# loop over protocols
# print(' ==== PROTOCOLS FROM NIDAQ DATA ====  ')
for iRec, protocol in enumerate(datatable['protocol']):
    _, _, onsets = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))
    print(' rec #%i) n=%i episodes, %s' % (iRec+1, len(onsets), protocol))
    DF.loc[iRec, 'daq-nEpisodes'] = len(onsets)

# %% [markdown]
#
# ## Load Open-Ephys data

# %%
sessions = []
for folder in datatable['Npx-Folder'].unique():
    sessions.append(\
        OpenEphysSession(os.path.join(datafolder, folder)))


# %%

print(' ==== PROTOCOLS FROM OPEN-EPHYS DATA ====  ')
props = []
iRec = 0
for session in sessions:
    for r, rec in enumerate(session.recordnodes[NODE].recordings):

        pulse_onsets, SN, TTL = load_OpenEphys(rec)

        fig, ax = pt.figure(axes=(1,2), ax_scale=(2.5, 1.5), hspace=0)
        fig.suptitle('Recording #%i' % (r+1))
        ax[1].set_xlabel('N, sample number (Npx Probe)')
        ax[0].set_ylabel('TTL (all)'); ax[1].set_ylabel('splitted')
        pt.plot(SN, TTL, ax=ax[0])

        # tracking different protocols
        # --> more than 2s between protocols to identify protocol changes
        iStarts = np.concatenate([[0], 
                                np.flatnonzero(np.diff(SN)>(30e3*INTERPROTOCOL_WINDOW)),
                                [len(SN)]])

        for i0, i1 in zip(iStarts[:-1], iStarts[1:]):

            irange=np.arange(i0, np.min([i1+2,len(SN)]))
            print(iRec, len(irange))
            if len(irange)>10:
                pulse_cond = (pulse_onsets>=SN[irange[0]]) & (pulse_onsets<=SN[irange[-1]])
            
                ax[1].plot(SN[irange], TTL[irange], lw=0.3, color=pt.tab10(iRec%10))
                pt.annotate(ax[1], 'protocol #%i'%(1+iRec) +iRec*'\n', (1,0), va='bottom', color=pt.tab10(iRec%10))

                DF.loc[iRec, 'i0'] = i0
                DF.loc[iRec, 'i1'] = i1
                DF.loc[iRec, 'Npx-Rec'] = 'node%i/exp%i/rec%i' % (NODE, EXP, r+1)
                DF.loc[iRec, 'ephys-nEpisodes'] = len(pulse_onsets[pulse_cond])

                iRec += 1

        pt.set_common_xlims(ax)
DF


# %%
_, _, fig, ephys_onsets, nidaq_onsets = sampling_match(11, 
                                                       datafolder, DF,
                                                       with_fig=True, verbose=True)


# %%
#
for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _, _, _ =\
            sampling_match(iRec, 
                           datafolder, DF,
                           with_fig=True)
DF

# %%

from physion.assembling.dataset import add_to_table

for key in ['Npx-Rec', 'nStart', 'nStop']:
    add_to_table(
        os.path.join(datafolder, 'DataTable0.xlsx'),
        sheet='Recordings',
        column=key,
        data=DF[key],
        insert_at=16 if 'nS' in key else 0)
DF
# %%
