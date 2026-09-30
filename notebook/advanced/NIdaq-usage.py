# %% [markdown]
# # Interface for NIdaq usage
#
# everything in SI units (seconds, volts, hertz)


# %% [markdown]
# ## Finding NI devices
#

# %%
import sys, time
sys.path += ['../../src']
from physion.hardware.NIdaq.config import (
    find_x_series_devices, find_m_series_devices, find_usb_devices
)
from physion.hardware.NIdaq.main import Acquisition

import numpy as np
import matplotlib.pylab as plt

devices = find_usb_devices()
if len(devices)>0:
    for device in devices:
        print('[ok] USB-series card found:', device)
devices = find_x_series_devices()
if len(devices)>0:
    for device in devices:
        print('[ok] X-series card found:', device)
devices = find_m_series_devices()
if len(devices)>0:
    for device in devices:
        print('[ok] M-series card found:', device)

# %% [markdown]
# ## See the available channels
# [!!] those are the channels from the API  
#
#   --> check each device property to see if available [!!]
#
# %%
from physion.hardware.NIdaq.config import (
    get_analog_input_channels, get_analog_output_channels,
    get_digital_input_channels, get_digital_output_channels
)
get_analog_input_channels(device)
get_digital_input_channels(device)

# %% [markdown]
# ## Run an Acquisition/Stim. with digital data
# map some outputs to inputs to check that things are working fine

# %%
import numpy as np
tstop = 4
DT = 0.25 # pulse at 4Hz in chan2
acq = Acquisition(
    sampling_rate=1000,
    # Nchannel_analog_in=1,
    max_time=tstop,
    digital_output_port="port0/line0:2",
    digital_input_port="port0/line3:7",
    digital_output_steps=\
        [{'channel':0, 'onset':t, 'duration':0.1}\
                 for t in np.arange(int(tstop/DT)+1)*DT]+\
        [{'channel':1, 'onset':t+DT/2., 'duration':0.1}\
                 for t in np.arange(int(tstop/DT)+1)*DT],
    filename='data.npy',
    verbose=True
)
acq.launch()
t0 = time.time()
while (time.time()-t0)<tstop:
    time.sleep(0.2)
acq.close()

import matplotlib.pylab as plt
fig, ax = plt.subplots(1, figsize=(6,2))
for i in range(acq.digital_data.shape[0]):
    ax.plot(1.1*i+acq.digital_data[i,:])

# %% [markdown]
# ## Run an Acquisition/Stim. with **analog** data
# map some outputs to inputs to check that things are working fine !
#
# output waveforms are best provided as functions of the time array `t`

# %%
import numpy as np

def fancy_time_func1(t):
    output = np.sin(2*np.pi*t)
    return output

def fancy_time_func2(t):
    output = np.cos(2*np.pi*t)
    return output

tstop = 4
DT = 0.25 # pulse at 4Hz in chan2
acq = Acquisition(
    sampling_rate=1000,
    Nchannel_analog_in=5,
    analog_output_funcs=[\
        fancy_time_func1,
        fancy_time_func2,
    ],
    max_time=tstop,
    verbose=True
)
acq.launch()
t0 = time.time()
while (time.time()-t0)<tstop:
    time.sleep(0.2)
acq.close()

import matplotlib.pylab as plt
fig, ax = plt.subplots(1, figsize=(6,2))
for i in range(acq.analog_data.shape[0]):
    ax.plot(acq.analog_data[i,:], label='AI%i' %i)
ax.legend(frameon=False, loc=(1,0.2))

# %%
# [markdown]
# ### Pulses


import numpy as np
import matplotlib.pylab as plt
tstop = 5.     # max time (seconds)
T = 30e-3       # period (seconds)
fs = 10e3       # sampling frequency (Hz)

t = np.arange(int(tstop*fs)+1)/fs

# %%

def build_start_stop_signal(t,
                            width=0.3):

    cond = ((t>0.1) & (t<(0.1+width))) |\
          ((t>(t[-1]-0.1-width)) & (t<(t[-1]-0.1)))
    output = np.zeros_like(t)
    output[cond] = 5
    return output

# %%

acq = Acquisition(\
                 sampling_rate=fs,
                 Nchannel_analog_in=3,
                 analog_output_funcs=[\
                     build_start_stop_signal,
                 ],
                 max_time=tstop)

acq.launch()
tic = time.time()
tac = tic
while (time.time()-tic)<tstop:
    if (time.time()-tac>2):
        print('    acq running t=%.1f' % (time.time()-tic))
        tac = time.time()
    pass
acq.close()

fig, ax = plt.subplots(1, figsize=(6,2))
for i in range(acq.analog_data.shape[0]):
    ax.plot(acq.analog_data[i,:], label='AI%i' %i)
ax.legend(frameon=False, loc=(1,0.2))

# %%
