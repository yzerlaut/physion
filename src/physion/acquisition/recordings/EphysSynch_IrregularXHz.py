"""


"""
import numpy as np

def synch_sequence(tmax,
                    freq = 2., # Hz
                      pulse_duration=0.1, # s
                        seed=10):

    np.random.seed(seed) 

    meanN = int(tmax*freq) # mean number of events
    interstim = 1./freq

    min_duration = np.max([2*pulse_duration, .5*interstim])

    if min_duration>=interstim:
        raise BaseException('\n [!!] Too long pulse duration in Ephys-Synch (given the freq.) \n ')

    interstim = np.random.uniform(interstim-min_duration,
                                  interstim+min_duration,
                                  size=2*meanN) #
    
    starts = np.cumsum(interstim)

    # should be a list to allow summation
    # return list(starts[starts<tmax])

    return starts[starts<tmax]

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


