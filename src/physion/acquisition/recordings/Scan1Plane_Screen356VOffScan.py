import numpy as np

###########################################
# ------------  2P trigger  ------------  #
###########################################
t0, length = 0.1, 0.1

def trigger2P(t):
    array = np.zeros(len(t), dtype=float)
    array[(t>=t0) & (t<(t0+length))] = 1.
    return 5.*array 

###########################################
# ----------  Off Scan Periods  --------  #
###########################################
framePeriod= 0.033986672351684885
duration = 1.2e-3
security = 0.1e-3

def off_scan_periods(t):
    array = np.zeros(len(t), dtype=float)
    array[ ( ((t-t0)%framePeriod)>(framePeriod-duration-security) ) &\
                 ( ((t-t0)%framePeriod)<(framePeriod-security) )  ] = 1.
    return array

output_funcs = [trigger2P,
                off_scan_periods]

