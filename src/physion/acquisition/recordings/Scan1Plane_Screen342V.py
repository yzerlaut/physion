import numpy as np

###########################################
# ------------  2P trigger  ------------  #
###########################################
2P_trigger_delay = 0.1 # second
2P_pulse_length = 0.1 # second

def trigger2P(t):
    array = np.zeros(len(t), dtype=float)
    array[(t>=2P_trigger_delay) & (t<(2P_trigger_delay+2P_pulse_length))] = 1.
    return 5.*array 

output_funcs = []
