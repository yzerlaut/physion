import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d
from physion.ephys.tools import compute_freq_envelope


##################################################
########### Processing of the LFP ################
##################################################

def heaviside(x):
    """ heaviside (step) function """
    return (np.sign(x)+1)/2


def compute_pLFP(LFP, sampling_freq,
                 freqs = np.linspace(40, 140, 20),
                 new_dt = None, # desired time subsampling 
                 subsample_before=True, # 
                 smoothing=40e-3):
    """
    performs continuous wavelet transform and smooth the time-varying high-gamma freq power
    """

    # compute the step corresponding to the desired subsampling freq
    if new_dt is not None:
        isubsmpl = int(new_dt*sampling_freq)
    else:
        isubsmpl=1

    if subsample_before:
        # computing the time-varying envelope
        W = compute_freq_envelope(LFP[::isubsmpl], sampling_freq/isubsmpl, freqs)
        # then smoothing
        pLFP = gaussian_filter1d(W, smoothing*sampling_freq/isubsmpl)

    else:
        # computing the time-varying envelope
        W = compute_freq_envelope(LFP, sampling_freq, freqs)
        # resampling and smoothing
        pLFP = gaussian_filter1d(np.reshape(W[:int(len(W)/isubsmpl)*isubsmpl],
                                            (int(len(W)/isubsmpl),isubsmpl)).mean(axis=1),
                                 int(smoothing/new_dt)).flatten()
        
    # insuring a time sampling matching those of the original data:
    return 1./sampling_freq*np.arange(len(LFP))[::isubsmpl][:len(pLFP)], pLFP
    

def NSI_func(low_freqs_envelope, sliding_mean,
             p0=0.,
             alpha=2.):
    """
    p0 should be the 100th percentile of the signal. It can be a sliding percentile.
    """
    X = (p0+alpha*low_freqs_envelope)-sliding_mean # rhythmicity criterion
    return -2*low_freqs_envelope*heaviside(X)+heaviside(-X)*(sliding_mean-p0)



def compute_sliding_mean(signal, sampling_freq,
                         T=500e-3):
    """ just a gaussian smoothing """
    return gaussian_filter1d(signal, int(T*sampling_freq))
    

def compute_NSI(signal, sampling_freq,
                p0=0,
                low_freqs = np.linspace(2,5,5),
                T_sliding_mean=500e-3,
                alpha=2.87,
                with_subquantities=False):
    """
    1. compute sliding-mean and low-freq envelope
    2. apply NSI formula
    """    
    sliding_mean = compute_sliding_mean(signal, sampling_freq, T=T_sliding_mean)
    
    low_freqs_envelope = compute_freq_envelope(signal, sampling_freq, low_freqs)

    if with_subquantities:
        return low_freqs_envelope, sliding_mean, NSI_func(low_freqs_envelope, sliding_mean,
                                                          p0=p0,
                                                          alpha=alpha)
    else:
        return NSI_func(low_freqs_envelope, sliding_mean,
                        p0=p0,
                        alpha=alpha)
        
    
    
def validate_NSI(t_NSI, NSI,
                 Tstate=200e-3,
                 var_tolerance_threshold=2):
    """
    iterates over episodes to perform state validation
    """
    # validate states:
    iTstate = int(Tstate/(t_NSI[1]-t_NSI[0]))
    NSI_validated = np.zeros(len(NSI), dtype=bool) # false by default

    # validate the transitions iteratively
    for i in np.arange(len(NSI))[::iTstate][1:-1]:
        if np.sum(np.abs(NSI[i-iTstate:i+iTstate]-NSI[i])>var_tolerance_threshold)==0:
            NSI_validated[i]=True # swith to true

    return NSI_validated


    
if __name__=='__main__':

    # ---  minimal working example (see README) --- #

    import numpy as np
    
    # -- let's build a fake LFP signal array (having the code features of an awake LFP signal)
    tstop, dt, sbsmpl_dt = 5, 1.2345e-4, 5e-3 # 10s @ 1kHz
    t = np.arange(int(tstop/dt))*dt
    oscill_part = ((1-np.cos(2*np.pi*3*t))*np.random.randn(len(t))+4*(np.cos(2*np.pi*3*t)-1))*\
        (1-np.sign(t-2))/2.*(2-t)/(tstop-2)
    desynch_part = (1-np.sign(2-t))/2*(t-2)/(tstop-2)*2*np.random.randn(len(t))
    LFP = (oscill_part+desynch_part)*.1 # a ~ 1mV ammplitude signal

    # -- compute the pLFP first
    t_pLFP, pLFP = compute_pLFP(1e3*LFP, 1./dt,
                                    freqs = np.linspace(50,300,10),
                                    new_dt=sbsmpl_dt,
                                    smoothing=42e-3)
    p0 = np.percentile(pLFP, 1) # first 100th percentile

    # -- then compute the NSI from the pLFP
    NSI = compute_NSI(pLFP, 1./sbsmpl_dt,
                          low_freqs = np.linspace(2, 5, 4),
                          p0=p0,
                          alpha=2.85)

    # then validate NSI episodes
    vNSI = validate_NSI(t_pLFP, NSI,
                            var_tolerance_threshold=20*p0) # here no noise so we increase the thresh

    
    # let's plot the result
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(3, 1, figsize=(8,4))
    ax[0].plot(t, LFP, color=plt.cm.tab10(7))
    ax[1].plot(t_pLFP, pLFP, color=plt.cm.tab10(5))
    ax[2].plot(t_pLFP, NSI, color=plt.cm.tab10(4), label='raw')
    ax[2].plot(t_pLFP[vNSI], NSI[vNSI], 'o', label='validated', lw=0, color=plt.cm.tab10(5))
    ax[2].legend(frameon=False)
    
    for x, label in zip(ax, ['LFP (mV)', 'pLFP (uV)', 'NSI (uV)']):
        x.set_ylabel(label)
        if 'NSI'in label:
            x.set_xlabel('time (s)')
        else:
            x.set_xticklabels([])
    # fig.savefig('doc/synthetic-example.png')
    plt.show()