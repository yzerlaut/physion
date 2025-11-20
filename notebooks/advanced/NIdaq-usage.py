# %%
import sys, time
sys.path += ['../../src']
import physion



# %%
import numpy as np
import matplotlib.pylab as plt
tstop = 0.5
T = 30e-3          # period (seconds)
fs = 10e3       # sampling frequency
dt = 1/fs

t = np.arange(int(tstop/dt))*dt

x = np.arange(len(t))

nT = int(T/dt)

cond = x % (3*nT) < nT

output = np.zeros_like(x)
output[cond] = 1

plt.plot(t, output)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# %%
acq = physion.hardware.NIdaq.main.Acquisition(\
                 sampling_rate=fs,
                 Nchannel_analog_in=2,
                 outputs=np.array([output], dtype=np.float64),
                 max_time=tstop)

# %%
acq.launch()
tic = time.time()
while (time.time()-tic)<tstop:
    pass
acq.close()
# %%
plt.plot(t, acq.analog_data[0][1:])
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# %%
