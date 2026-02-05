"""


"""
import numpy as np

def synch_sequence(tmax,
                   Freq = 5., # Hz
                   pulse_duration=10e-3,
                   seed=10):

    np.random.seed(seed) 

    meanN = int(tmax*Freq) # mean number of events
    interstim = 1./Freq

    interstim = np.random.uniform(.5*interstim,
                               1.5*interstim,
                               size=2*meanN) #
    
    starts = np.cumsum(interstim)

    # should be a list to allow summation
    return list(starts[starts<tmax])

    # PREVIOUS VERSION WITH digital_output_funcs
    # array = np.zeros(len(t), dtype=float) # 1 by default
    # for start in starts:
    #     if start<t[-1]:
    #         cond = ( t>= start ) & ( t< (start + pulse_duration) )
    #         array[cond] = cmd_level

    # return array


if __name__=='__main__':
    seq = synch_sequence(10)
    for s in seq:
        print(s)
    print('--> n=%i' % len(seq))


