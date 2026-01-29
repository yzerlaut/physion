"""


"""
import numpy as np

def synch_sequence(t, stim, cmd_level,
                   Freq = 5., # Hz
                   pulse_duration=10e-3,
                   seed=10):

    np.random.seed(seed) 

    meanN = int(t[-1]*Freq) # number of 
    interstim = 1./Freq

    starts = np.random.uniform(.5*interstim,
                               1.5*interstim,
                               size=2*meanN) #

    array = np.zeros(len(t), dtype=float) # 1 by default
    for start in starts:
        if start<t[-1]:
            cond = ( t>= start ) & ( t< (start + pulse_duration) )
            array[cond] = cmd_level

    return array

output_funcs = [synch_sequence]


if __name__=='__main__':
    print(3)


