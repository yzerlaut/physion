
# %% [markdown]
#
# # Assemble Neuropixels data
#
# requirements:
# ```
# pip install open-ephys-python-tools
# ```
# ## 1) 
# Run:
# ```
# python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA\2026_02_13
# ```
# this will create a file: `%USERPROFILE%\DATA\DataTable0.xlsx`  
#      move it to  ~/DATA/2026_02_13/DataTable0.xlsx
#
# Then fill its neuropixels folder (`Npx-Folder`) and recordings information (`Npx-Rec`).    
#
#       N.B. you can use the code below to guide filling the recordings info

# %%
import sys, time
sys.path += [os.path.expanduser('~/physion/src')]
import json
import numpy as np
from open_ephys.analysis import Session

from physion.assembling.dataset import read_spreadsheet
from physion.acquisition.tools import find_line_props
import physion.utils.plot_tools as pt
pt.set_style('dark')

# %% [markdown]
#
# ## Load Table data

# %%
datafolder = os.path.expanduser(\
                '~/DATA/2026_02_13').replace('/', os.path.sep)

datatable, _, analysis = read_spreadsheet(\
                        os.path.join(datafolder, 'DataTable0.xlsx'),
                                   get_metadata_from='files')
datatable

# %% [markdown]
#
# ## Load NIdaq data

# %%
#
def load_nidaq_synch_signal(folder):
    """ """
    with open(os.path.join(folder, 'metadata.json')) as f:
        metadata = json.load(f)
    NIdaq = np.load(os.path.join(folder, 'NIdaq.npy'),
                    allow_pickle=True).item()
    props = find_line_props(
                metadata['NIdaq']['digital-outputs']['line-labels'])
    ephysSynch_signal = NIdaq['digital'][props['chan']]
    nSteps = len(np.flatnonzero(ephysSynch_signal[1:]>ephysSynch_signal[:-1]))
    t = np.arange(len(ephysSynch_signal))*NIdaq['dt']
    return t, ephysSynch_signal, nSteps

# loop over protocols
print(' ==== PROTOCOLS FROM NIDAQ DATA ====  ')
for iRec, protocol in enumerate(datatable['protocol']):
    _, _, nSteps = load_nidaq_synch_signal(
                        os.path.join(datafolder, datatable['time'][iRec]))
    print(' rec #%i) n=%i episodes, %s' % (iRec+1, nSteps, protocol))


# %% [markdown]
#
# ## Load Open-Ephys data


# %%

INTERPROTOCOL_WINDOW = 10. # 

node = 0 # change if you have several record nodes and you want to consider another one

session = Session(os.path.join(datafolder, 
                               datatable['Npx-Folder'][0]))

print(' ==== PROTOCOLS FROM OPEN-EPHYS DATA ====  ')
props = []
iRec = 0
for r, rec in enumerate(session.recordnodes[node].recordings):

    fig, ax = pt.figure(axes=(1,2), ax_scale=(2.5, 1.5), hspace=0)
    fig.suptitle('Recording #%i' % (r+1))
    ax[1].set_xlabel('N, sample number (Npx Probe)')
    ax[0].set_ylabel('TTL (all)'); ax[1].set_ylabel('splitted')

    # find TTL events on Probe A
    cond = (rec.events['stream_name']=='ProbeA')

    # build the time array from the set of events
    State = np.array(rec.events['state'][cond])
    Sample = np.array(rec.events['sample_number'][cond])
    SN, TTL = [Sample[0]-30000], [0]
    for state, sample in zip(State, Sample):
        if state==1:
            SN.append(sample); TTL.append(0)
            SN.append(sample); TTL.append(1)
        if state==0:
            SN.append(sample); TTL.append(1)
            SN.append(sample); TTL.append(0)
    SN.append(sample+30000); TTL.append(0)
    SN, TTL = np.array(SN, dtype=np.int32), np.array(TTL, dtype=np.uint8)
    pt.plot(SN, TTL, ax=ax[0])

    # tracking different protocols
    # --> more than 2s between protocols to identify protocol changes
    iStarts = np.concatenate([[0], 
                              np.flatnonzero(np.diff(SN)>(30e3*INTERPROTOCOL_WINDOW)),
                              [len(SN)]])

    for i0, i1 in zip(iStarts[:-1], iStarts[1:]):

        irange=np.arange(i0, np.min([i1+2,len(SN)]))
        props.append({'node':node, 'rec':r, 
                        'i0':i0, 'i1':i1,
                        'sn':SN[irange], 'ttl':TTL[irange]})
        props[-1]['nsteps']=np.sum((props[-1]['ttl'][1:]==1)&(props[-1]['ttl'][:-1]==0))
        iRec +=1
        print(' rec #%i) n=%i episodes' % (iRec, props[-1]['nsteps']))

        ax[1].plot(props[-1]['sn'], props[-1]['ttl'], color=pt.tab10(iRec%10))
        pt.annotate(ax[1], 'protocol #%i'%iRec +iRec*'\n', (1,0), va='bottom', color=pt.tab10(iRec%10))

    pt.set_common_xlims(ax)


                  
# %%
# VISUALIZE THE TWO SIGNALS

iRec = 1

fig, AX = pt.figure(axes=(2,2), ax_scale=(1.5,1), hspace=1.4, wspace=0.4, top=.5)
# nidaq
t, ephysSynch_signal, _ = load_nidaq_synch_signal(
                            os.path.join(datafolder, datatable['time'][iRec]))
AX[0][0].plot(t[:10000:10], ephysSynch_signal[:10000:10])
AX[0][1].plot(t[-10000:][::10], ephysSynch_signal[-10000:][::10])
for i, ax in enumerate(AX[0]):
    pt.set_plot(ax, xlabel='NIdaq time (s)', ylabel='TTL\n(from NIdaq)' if i==0 else None)

# open-ephys
AX[1][0].plot(props[iRec]['sn'][1:70], props[iRec]['ttl'][1:70], 'o-', lw=0.4, ms=0.9)
AX[1][1].plot(props[iRec]['sn'][-70:-1], props[iRec]['ttl'][-70:-1], 'o-', lw=0.4, ms=0.9)
for i, ax in enumerate(AX[1]):
    pt.set_plot(ax, xlabel='N, sample number (Npx Probe)      ', ylabel='TTL\n(on Probe)' if i==0 else None)
# fig.savefig(os.path.expanduser('~/Desktop/fig.png'))
fig.savefig('fig.png')


# %%
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.optimize import least_squares


def find_sampling_match(t, nidaqTTL, sn, ttl):
    """
    ADD SUBSAMPLING OF TIME ARRAY FOR THE FIT
    """

    N0 = sn[1] # - 3000
    t0 = t[np.flatnonzero(nidaqTTL==1)][0]
    
    # dN = np.median(np.diff(sn[1::4]))
    # dT = np.mean(np.diff(t[1:][nidaqTTL[1:]>nidaqTTL[:-1]]))


    dN = sn[-4]-sn[1]
    nidaqJump = np.flatnonzero(nidaqTTL[1:]>nidaqTTL[:-1])
    dT = t[nidaqJump[-1]]-t[nidaqJump[0]]

    F0 = dT/dN

    # def to_minimize(x):
    #     T = (sn-x[0])*x[1]+t0
    #     func = interp1d(T, ttl) 
    #     probe_signal = func(np.clip(t, T.min(), T.max()))
    #     return np.mean((probe_signal-nidaqTTL)**2)

    # res = least_squares(to_minimize, [N0, F0],
    #                     # max_nfev=10000, method='dogbox',
    #                     # ftol=None, xtol=None, verbose=True,
    #                     bounds=[(N0-3000, 0.99*F0), (N0+3000, 1.01*F0)])
    # N0, F0 = res.x

    return t0, N0, F0

def sampling_match(iRec,
                   with_fig=False):

    t, ephysSynch_signal, nSteps = load_nidaq_synch_signal(
                                os.path.join(datafolder, datatable['time'][iRec]))

    t0, N0, F0 = find_sampling_match(t, ephysSynch_signal, props[iRec]['sn'], props[iRec]['ttl'])

    T = (props[iRec]['sn']-N0)*F0+t0 # new time sampling
    func = interp1d(T, props[iRec]['ttl'])
    wide_t = -0.5+np.arange(len(t)+20000)*t[1]
    wide_t = wide_t[wide_t<T[-1]]

    probe_signal = func(wide_t)

    width=1

    if with_fig:

        fig, AX = pt.figure(axes=(4,2), ax_scale=(1.6,.7), top=1.5, hspace=1.4, wspace=0.3)
        fig.suptitle('protocol #%i (%i episodes)' % (iRec+1, nSteps))

        for i, t0 in enumerate([0.5, t[-1]/2+1, 3.*t[-1]/4., t[-1]]):

            pt.annotate(AX[0][i], 't=%.1fs' % t0, (0.1,1))
            # nidaq
            cond = (t>(t0-width)) & (t<(t0+width))
            AX[0][i].plot(t[cond][::10], ephysSynch_signal[cond][::10])
            pt.set_plot(AX[0][i], xlabel='NIdaq time (s)', ylabel='TTL\n(from NIdaq)' if i==0 else None)

            # open-ephys
            cond = (wide_t>(t0-width)) & (wide_t<(t0+width))
            AX[1][i].plot(wide_t[cond][::10], probe_signal[cond][::10])
            pt.set_plot(AX[1][i], xlabel='$F \\cdot (N- N_0) $ time (s)', ylabel='TTL\n(on Probe)' if i==0 else None)

            pt.set_common_xlims([AX[0][i], AX[1][i]])

        return t0, N0, F0, fig
    else:
        return t0, N0, F0

sampling_match(7, with_fig=True)

# %%
#
for iRec, time in enumerate(datatable['time']):

    sampling_match(iRec, with_fig=True)


# %%
fig, AX = pt.figure(axes=(1,2), ax_scale=(2,1), hspace=1.4, top=.5)
# nidaq
t, ephysSynch_signal, _ = load_nidaq_synch_signal(
                            os.path.join(datafolder, datatable['time'][iRec]))
AX[0].plot(t[::100], ephysSynch_signal[::100])
pt.set_plot(AX[0], xlabel='NIdaq time (s)', ylabel='TTL\n(from NIdaq)')
# open-ephys
AX[1].plot(props[iRec]['sn'], props[iRec]['ttl'], 'o-', lw=0.4, ms=0.9)

# %%

def find_openephys_in(dayfolder, verbose=True):

    day = dayfolder.split(os.path.sep)[-1].replace('_', '-')
    # print(day, os.listdir(dayfolder))
    folders = [f for f in os.listdir(dayfolder) if day in f]
    if len(folders)==1:
        return os.path.join(dayfolder, folders[0])
    else:
        print()
        print('  [!!] Not a single open-Ephys recording found [!!]')
        print('             list of folders:', folders)
        print()

session = Session(find_openephys_in(root_datafolder))

# %%
# for rec in session.recordnodes[0].recordings:
    # print(rec.events)
rec = session.recordnodes[0].recordings[0]

condUp = (rec.events['stream_name']=='ProbeA') & (rec.events['state']==1)
condDown = (rec.events['stream_name']=='ProbeA') & (rec.events['state']==0)

rec.events['timestamp']

iStarts = np.concatenate([[0], np.flatnonzero(np.diff(rec.events['timestamp'])>2.0), [-1]]) # more than two seconds between protocols
print(len(iStarts))

# %%
sync_samples = []
for i0, i1 in zip(iStarts[:-1], iStarts[1:]):
    sync_samples.append(sn[i0:i1][::2])

for s0, s1 in zip(condUp)
ttlEphys = np.zeros(len())
# start from timestamps
ts = rec.events['timestamp'][cond]
sn = rec.events['sample_number'][cond]
# convert to intervals + s --> ms
intervals = np.diff(sn)
intervals 
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter1d
# plt.hist(intervals[:200])
plt.plot(gaussian_filter1d(intervals[:150], 10))
plt.yscale('log')
# print(\
#      rec.events['timestamp'][cond])

# rec.add_sync_line(1,            # TTL line number
#                     100,          # processor ID
#                         'ProbeA',   # stream name
#                             main=True)   # align to the main stream
# rec.compute_global_timestamps()
# print(rec.sync_lines)

# %%

def load_nidaq_and_metadata(folder):
    """ """
    with open(os.path.join(folder, 'metadata.json')) as f:
        metadata = json.load(f)
    NIdaq = np.load(os.path.join(folder, 'NIdaq.npy'),
                    allow_pickle=True).item()
    return metadata, NIdaq

for day, time in zip(datatable['day'], datatable['time']):
    metadata, NIdaq = load_nidaq_and_metadata(os.path.join(root_datafolder, time))
    props = find_line_props(
                metadata['NIdaq']['digital-outputs']['line-labels'])
    ephysSynch_signal = NIdaq['digital'][props['chan']]
    steps = np.flatnonzero(ephysSynch_signal[1:]>ephysSynch_signal[:-1])
    print(len(steps))
    # with open(os.path.join(folder, 'NIdaq.npy')) as f:
    #     metadata = json.load(f)

# %%
np.unique(datatable['Npx-Rec'])

# %%
import os
import numpy as np

dayfolder = os.path.expanduser('~/DATA/2026_02_13').replace('/', os.path.sep)


def extract_ephys_string(rec):
    node = int(rec.split('/')[0].replace('node',''))
    exp = int(rec.split('/')[1].replace('exp',''))
    rec = int(rec.split('/')[2].replace('rec',''))
    return node, exp, rec

def find_set_of_sample_numbers(folder):

    folder = os.path.join(folder, 
                          'events', 'OneBox-100.ProbeA', 'TTL')
    
    if os.path.isdir(folder):
        ts = np.load(os.path.join(folder, 'timestamps.npy'))
        sn = np.load(os.path.join(folder, 'sample_numbers.npy'))
        
        iStarts = np.concatenate([[0], np.flatnonzero(np.diff(ts)>2.0), [-1]]) # more than two seconds between protocols

        sync_samples = []
        for i0, i1 in zip(iStarts[:-1], iStarts[1:]):
            sync_samples.append(sn[i0:i1][::2])

        return sync_samples
    else:
        return None

def load(dayfolder, 
         rec='node101/exp1/rec1'):

    ephysFolder = os.path.join(datafolder, datatable['Npx-Folder'][0]) 
    node, exp, rec = extract_ephys_string(rec)

    rec_folder = os.path.join(dayfolder, 
                          ephysFolder,
                          'Record Node %i' % node,
                          'experiment%i' % exp,
                          'recording%i' % rec)

    if os.path.isdir(rec_folder):
        return find_set_of_sample_numbers(rec_folder)
    else:
        return None

# %%


# %%
import itertools

nProtocols = 0
protocols_in_ephys = []

for node, exp, rec in itertools.product([101], range(1, 5), range(1, 10)):

    sync_samples = load(dayfolder, 'node%i/exp%i/rec%i' % (node, exp, rec))
    if sync_samples is not None:
        for i in range(len(sync_samples)):
            print(' - node%i/exp%i/rec%i ' % (node, exp, rec), ' -->  n=%i synch. pulses' % len(sync_samples[i]))
        nProtocols += len(sync_samples)

print(nProtocols)
# sync_samples = load(dayfolder, 'node101/exp1/rec1')

# %%


f = "C:\\Users\\info\\OneDrive\\Documents\\Open Ephys\\2026-02-03_16-04-25\\Record Node 103\\experiment1\\recording1\continuous\OneBox-102.OneBox-ADC\\timestamps.npy"
f = "C:\\Users\\info\\OneDrive\\Documents\\Open Ephys\\2026-02-03_16-04-25\\Record Node 103\\experiment1\\recording1\events\MessageCenter\\timestamps.npy"

folder = "C:\\Users\info\DATA\\2026-02-03_17-47-23\Record Node 101\experiment1\\recording1\events\OneBox-100.ProbeA\TTL"
print(' timestamps: ',  np.load(os.path.join(folder, 'timestamps.npy')))
print(' sample numbers: ',  np.load(os.path.join(folder, 'sample_numbers.npy')))

# %%
fig, ax = plt.subplots(1, figsize=(6,2))
for i in range(acq.digital_data.shape[0]):
    ax.plot(1.1*i+acq.digital_data[i,:])
# ax.plot(self.digital_data[0,:])

# %%
# zoom properties
plt.plot(1e3*t[::10], acq.analog_data[0][::10], label='chan0')
plt.plot(1e3*t[::10], acq.analog_data[1][::10], label='chan1')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend(loc=(1.,0.2))

# %%
# now ZOOM on data
t0, width =0.1, 0.1
cond = (t>t0) & (t<(t0+width))
plt.plot(1e3*(t[cond]-t0), acq.analog_data[0][cond], label='start')
plt.plot(1e3*(t[cond]-t0), acq.analog_data[1][cond], label='stop')
plt.xlabel("Time (ms)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend(loc=(1.,0.2))

# %%
# performing multiple recordings

for i in range(1,4):
        
    fs = 50e3
    tstop = 3 # 60*60
    t = np.arange(int(tstop*fs)+1)/fs
    output = build_start_stop_signal(t)
    acq = Acquisition(\
                sampling_rate=fs,
                Nchannel_analog_in=2,
                outputs=np.array([output], dtype=np.float64),
                filename=os.path.expanduser('~/Desktop/Sample%i.npy' % i),
                max_time=tstop)
    acq.launch()
    tic = time.time()
    while (time.time()-tic)<tstop:
        pass
    acq.close()
    time.sleep(4)

import matplotlib.pylab as plt
fig, ax = plt.subplots(1, figsize=(6,2))
for i in range(acq.analog_data.shape[0]):
    ax.plot(acq.analog_data[i,:], label='AI%i' %i)
ax.legend(frameon=False, loc=(1,0.2))

# %%
import os
fig, ax = plt.subplots(2, figsize=(7,4))
for i in range(1, 4):
        
    data = np.load(\
        os.path.expanduser('~/Desktop/Sample%i.npy' % i),
        allow_pickle=True).item()
    t = np.arange(len(data['analog'][0]))*data['dt']
    t0 =0.132
    cond = (t>t0) & (t<(t0+0.003))

    ax[0].plot(1e3*(t[cond]-t0), data['analog'][0][cond], label='start')
    ax[1].plot(1e3*(t[cond]-t0), data['analog'][1][cond], label='stop')
    #plt.xlabel("Time (ms)")
    #plt.ylabel("Amplitude")
    plt.grid(True)
#plt.legend(loc=(1.,0.2))
# %%
data
# %%