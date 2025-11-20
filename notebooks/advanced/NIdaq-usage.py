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

A = np.zeros_like(x)
A[cond] = 1

plt.plot(t, A)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
# %%
