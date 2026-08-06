
# %% [markdown]
#
# # Assemble Neuropixels data, see [ephys doc](https://github.com/yzerlaut/physion/blob/main/src/physion/ephys/README.md)
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```

# %%
import sys, os
sys.path += [os.path.expanduser('~/physion/src'), '../src']
import numpy as np
import pandas as pd

from open_ephys.analysis import Session as OpenEphysSession
from physion.assembling.dataset import read_spreadsheet, add_to_table

################################################
###   code to do the alignement !!      ########
################################################
from physion.ephys.alignement import load_nidaq_synch_signal,\
    load_OpenEphys, sampling_match

import spikeinterface.full as si

import physion.utils.plot_tools as pt
pt.set_style('dark')

datafolder = os.path.expanduser('~/DATA/2026_08_04').replace('/', os.path.sep)

INTERPROTOCOL_WINDOW = 10. # 
ELECTRODE_RANGE = [0,200]
PROBE_NAME = 'ProbeA'
STREAM_NAME='Record Node 101#OneBox-100.%s' % PROBE_NAME
EXP = 1 # 
NODE=0

# %% [markdown]
#
# ## Load Table data

# %%

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable.xlsx'),
                                   get_metadata_from='files')
#datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
DF = pd.DataFrame(columns=['time', 
                           'daq-nEpisodes', 'ephys-nEpisodes', 
                           'i0', 'i1', 'nStart', 'nStop'])
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
# iRec = 0 # example recording
# _, _, fig, ephys_onsets, nidaq_onsets = sampling_match(iRec, 
#                                                        datafolder, DF,
#                                                     #    fitting_window=[0.2, 0.4],
#                                                        debug_fitting=True,
#                                                        with_fit_fig=True,
#                                                        with_fig=True)


# %%
#
#######################################################################
############ WRITING ALIGNEMENT SAMPLES in DataTable ##################
#######################################################################

for iRec, time in enumerate(datatable['time']):

    DF.loc[iRec, 'nStart'], DF.loc[iRec, 'nStop'], _, _, _ =\
            sampling_match(iRec, 
                           datafolder, DF,
                           with_fig=True)
# DF

# # %%
for key in ['Npx-Rec', 'nStart', 'nStop']:
    add_to_table(
        os.path.join(datafolder, 'DataTable.xlsx'),
        sheet='Recordings',
        column=key,
        insert_at=16,
        data=DF[key])
DF

# %%
#######################################################################
############ WRITING ELECTRODE RANGE AND BAD CHANNELS #################
#######################################################################

bad_channels = {}
for folder in np.unique(datatable['Npx-Folder']):

    siRec = si.read_openephys(os.path.join(datafolder, folder),
                            stream_name=STREAM_NAME)
    
    siRec = siRec.select_channels(\
        siRec.get_channel_ids()[np.arange(*ELECTRODE_RANGE)])

    bad_channel_ids, _ = si.detect_bad_channels(siRec,\
                                             method="coherence+psd")

    if len(bad_channel_ids)>0:
        bad_channels[folder] = bad_channel_ids[0]
        for c in bad_channel_ids[1:]:
            bad_channels[folder] += ','+c
    else:
        bad_channels[folder] = ''


# %%
add_to_table(
    os.path.join(datafolder, 'DataTable.xlsx'),
    sheet='Recordings', column='electrode-range',
    data=['%i-%i' % (ELECTRODE_RANGE[0], ELECTRODE_RANGE[1]) for _ in range(len(DF['Npx-Rec']))])

add_to_table(
    os.path.join(datafolder, 'DataTable.xlsx'),
    sheet='Recordings', column='electrode-subsampling',
    insert_at=6,
    data=[8 for _ in range(len(DF['Npx-Rec']))])

add_to_table(
    os.path.join(datafolder, 'DataTable.xlsx'),
    sheet='Recordings', column='bad-channels',
    insert_at=7,
    data=[bad_channels[f] for f in DF['Npx-Folder']])
# %%
