"""
"""
import numpy.fft as fft
import numpy as np
from scipy import signal

def filter(data, Facq,
              freq=300.0,
              fType='low-pass',
              order=5,
              axis=-1):
    """
    spectral filtering relying on scipy.signal methods
    doing forward-backward filtering ("filtfilt") to avoid phase shifts (e.g. delay with low-pass)
    args:

        fType: filter type, low-pass filtering, 'high', 'band'
    """
    sos = filter_kernel(Facq, fType=fType, freq=freq, order=order)
    return signal.sosfiltfilt(sos, data, axis=axis)

def filter_kernel(Facq,
                  freq=300.0,
                  fType='low-pass',
                  order=5,
                  output='sos'):
    """
        fType: filter type, low-pass filtering, 'high', 'band'
    """
    nyq = 0.5*Facq
    if type(freq)==list:
        freq = np.array(freq)
    return signal.iirfilter(order, freq/nyq, 
                            btype=fType.replace('-pass',''), 
                            analog=False, output=output)



if __name__=='__main__':

    import matplotlib.pyplot as plt
    
    # plotting
    fig = plt.figure(figsize=(8,5))
   # Create a 2-row grid: top row has 3 panels, bottom row has 1 wide panel
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
    axB = fig.add_subplot(gs[1, :])

    # # First make some data to be filtered.
    T = 1         # seconds
    fs = 1e3 # sample rate, Hz
    n = int(T * fs) # total number of samples
    t = np.linspace(0, T, n, endpoint=False)
    # "Noisy" data + harmonics
    data = np.random.randn(len(t))+2
    FREQS = [1.5, 15., 50.] # freq of harmonics
    for i, freq, amp in zip(range(3), FREQS, [2, 1, 1, 1]):
        random_phase = np.random.uniform(0, 2*np.pi)
        data += np.sin(freq*2*np.pi*t+random_phase)
        axB.plot(t, np.sin(freq*2*np.pi*t+random_phase), ':', color=plt.cm.tab10(i))

    axB.plot(t, data, 'k-')
    axB.set_xlabel('time (s)')
    axB.set_title('signal from harmonics @ %sHz' % str(FREQS))

    # Get the filter coefficients so we can check its frequency response.
    for i, fType, freq in zip(range(3), ['low', 'high', 'band'], [5, 20, np.array([10,20])]):

        ax = fig.add_subplot(gs[0, i])
        b, a = filter_kernel(fs, freq=freq, fType=fType, output='ba')
        w, h = signal.freqz(b, a)#, worN=8000)
        ax.plot(0.5*fs*w/np.pi, np.abs(h), color=plt.cm.tab10(i), lw=2)
        ax.set_title(" %s-pass filter: %sHz" % (fType, str(freq)), color=plt.cm.tab10(i))
        ax.set_xlabel('Frequency [Hz]')
        ax.set_xlim([0,50])
        ax.plot(freq, 0.5*np.sqrt(2)+0*freq, 'ko')
        if type(freq) is np.ndarray:
            for f in freq:
                ax.axvline(f, color='k')
        else:
            ax.axvline(freq, color='k')

        filtered = filter(data, fs, freq=freq, fType=fType)
        axB.plot(t, filtered, color=plt.cm.tab10(i), lw=1)

    # plt.xlim(0, 0.5*fs)
    # plt.title(filtertype+" Filter Frequency Response")
    # plt.xlabel('Frequency [Hz]')

    # # Demonstrate the use of the filter.
    # # First make some data to be filtered.
    # T = 5.0         # seconds
    # n = int(T * fs) # total number of samples
    # t = np.linspace(0, T, n, endpoint=False)
    # # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
    # data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)+5.+3*np.sin(cutoff*2*np.pi*t)

    # # Filter the data, and plot both the original and filtered signals.
    # y = signal.lfilter(b, a, data, axis=-1)

    # plt.subplot(2, 1, 2)
    # plt.plot(t, data, 'b-', label='data')
    # plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
    # plt.xlabel('Time [sec]')
    # plt.legend()

    plt.subplots_adjust(left=0.1, bottom=0.1, hspace=0.35)
    plt.show()