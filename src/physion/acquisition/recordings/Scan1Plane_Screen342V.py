import numpy as np

###########################################
# ------------  2P trigger  ------------  #
###########################################
TwoP_trigger_delay = 0.1 # second
TwoP_pulse_length = 0.1 # second

def trigger2P(t):
    array = np.zeros(len(t), dtype=float)
    array[(t>=TwoP_trigger_delay) & (t<(TwoP_trigger_delay+TwoP_pulse_length))] = 1.
    return 5.*array 

output_funcs = []
