# Electrophysiology

## Overall pipeline 

*following a given experiment recorded with OpenEphys & physion*

0. (prelim) Preprocess the facecamera data (Pupil & Facemotion)

1. Build the DataTable for the different protocols of the session

Open the terminal (miniforge) on the `base` environment:
```
cd physion/src
python -m physion.assembling.dataset build-DataTable %USERPROFILE%\DATA
```

run the [Ephys-Assembling.py](../../../notebooks/Ephys-Assembling.py) notebook.    
    This will:
    - detect and remove bad channels
    - define the range of intersting range of electrodes for (those in the brain)
    - compress and save as binary the data
    - generate a `DataTable.xlsx` file used for later assembling

2. run the spike sorting using spike interface (kilosort for now)

3. assemble the data into a NWB file using the command:  
`python -m physion.assembling.nwb ~/DATA/2026_01_01/DataTable.xlsx`


See: [../assembling/add_ephys.py](../assembling/add_ephys.py) script

```
import spikeinterface.full as si
```

1. remove bad channels (with `si.detect_bad_channels(rec, method="coherence+psd")`)
2. build the electrode table
3. add the spike times of units selected by `kilosort`/`phy`
4. add the spike templates of units selected by `kilosort`/`phy`
5. runs peak detection (locally exclusive, see `from spikeinterface.sortingcomponents import peak_detection`)
6. write the events per channel (of the above peak detection)
7. computes and write LFP

## Extracting Single Units Spiking

- spike sorting with kilosort

- 

## Computing the Local Field Potential (LFP)

- we band-pass filter in the band `[0.5, 300]`Hz

- 

N.B. Here we take care of not creating artefacts on boudaries of data chunks (so we increase the chunk_size to a really high value)

## Computing Multi-Unit Activity (MUA)

using the function `compute_freq_envelope` of `physion.ephys.tools` with a band in `[300,3000]`Hz.

This is using a continuous wavelet transform to extract the time-varying high-frequency activity in the recording.

To visualize the effect of the continuous wavelet transform acting in different frequency bands:

![envelope demo](../../docs/ephys/wavelet-envelope.png)
```
from physion.ephys.tools import compute_freq_envelope

t = np.linspace(0, 11, int(5e4))
signal = np.zeros(len(t))

np.random.seed(11)
N = 20
for freq, start, amp in zip(
    list(np.random.uniform(0.5, 20, N)), list(1+np.arange(N)/N*9), list(np.random.randn(N)),
    ):
    sigma = 1./freq/2.
    signal += np.sin(2*np.pi*freq*(t-start))*\
        np.exp(-(t-start)**2/2./sigma**2)*\ amp

fig, ax = pt.figure(axes=(1,4), ax_scale=(2,1.5))
ax[0].plot(t, signal)
pt.set_plot(ax[0], xlabel='time (s)', ylabel='signal (a.u.)')
for i, band in enumerate([[5,20], [1,10], [0.2,1]]):
    mua = compute_freq_envelope(signal, 1./(t[1]-t[0]),
                    np.linspace(band[0], band[1], 5))
    ax[i+1].fill_between(t, 0*t, mua)
    pt.set_plot(ax[i+1], xlabel='time (s)', ylabel='envelope (a.u.)',
                    title='freq. band: [%.1f,%.1f]Hz' % tuple(band) +\
                        40*'  ')
fig.savefig('physion/docs/ephys/wavelet-envelope.png')
```

