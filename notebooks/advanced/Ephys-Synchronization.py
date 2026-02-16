
# %%
# [markdown]

```
python -m physion.assembling.dataset build-DataTable ..\..\..\DATA\2026_02_13
```

# %%
import sys, time
sys.path += ['../../src']
from physion.assembling.dataset import read_spreadsheet

# %%
datatable, _, _ = read_spreadsheet(os.path.expanduser('~/DATA/DataTable0.xlsx'))
datatable

# %%
import os
import numpy as np

root_datafolder = os.path.expanduser('~/DATA')

def load(day,
         node='Record Node 101',
         exp='experiment5',
         rec='recording2'):
    
    folder = os.path.join(root_datafolder, day, node, exp, rec,
                    #   'events', 'MessageCenter')
                    #   'events', 'OneBox-100.OneBox-ADC', 'TTL')
                      'events', 'OneBox-100.ProbeA', 'TTL')
                    #   'continuous', 'OneBox-100.ProbeA')
    
    ts = np.load(os.path.join(folder, 'timestamps.npy'))
    sn = np.load(os.path.join(folder, 'sample_numbers.npy'))
    
    return ts, sn

ts, sn = load('2026-02-11_14-32-51')
print(len(ts))
print(len(sn))

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