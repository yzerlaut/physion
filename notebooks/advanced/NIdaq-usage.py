# %% [markdown]
### Be sure to pick the 


# %%
import sys, time
sys.path += ['../../src']
from physion.hardware.NIdaq.main import Acquisition

import numpy as np
import matplotlib.pylab as plt
tstop = 10.     # max time (seconds)
T = 30e-3       # period (seconds)
fs = 10e3       # sampling frequency (Hz)
dt = 1/fs

t = np.arange(int(tstop/dt)+1)*dt

# %%

def build_start_stop_signal(t):

    cond = ((t>0.1) & (t<0.15)) |\
          ((t>(t[-1]-0.1)) & (t<(t[-1]-0.05)))
    output = np.zeros_like(t)
    output[cond] = 5
    return output
output = build_start_stop_signal(t)

plt.plot(t[::10], output[::10])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# %%
acq = Acquisition(\
                 sampling_rate=fs,
                 Nchannel_analog_in=2,
                 outputs=np.array([output],
                                  dtype=np.float64),
                 max_time=tstop)

# %%
acq.launch()
tic = time.time()
tac = tic
while (time.time()-tic)<tstop:
    if (time.time()-tac>2):
        print('    acq running t=%.1f' % (time.time()-tic))
        tac = time.time()
    pass
acq.close()

# %%
# zoom properties
plt.plot(1e3*t[::10], acq.analog_data[0][::10], label='start')
plt.plot(1e3*t[::10], acq.analog_data[1][::10], label='stop')
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
