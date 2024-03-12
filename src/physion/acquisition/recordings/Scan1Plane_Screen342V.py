import numpy as np

###########################################
# ------------  2P trigger  ------------  #
###########################################
t0, length = 0.1, 0.1

def trigger2P(t):
    array = np.zeros(len(t), dtype=float)
    array[(t>=t0) & (t<(t0+length))] = 1.
    return 5.*array 

output_funcs = []
