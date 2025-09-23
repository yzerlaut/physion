import numpy as np

from . import Scan1Plane_Screen342V

###########################################
# ----------  Off Scan Periods  --------  #
###########################################
framePeriod= 0.033986672351684885
duration = 3.1e-3
security = 0.1e-3

TwoP_trigger_delay = Scan1Plane_Screen342V.TwoP_trigger_delay

nLines = 512

linePeriod = np.float64(framePeriod-duration)/np.float64(nLines)

def off_scan_periods(t):
    array = np.ones(len(t), dtype=float) # 1 by default
    # interframe space
    array[ ( ((t-TwoP_trigger_delay)%framePeriod)>(framePeriod-duration-security) ) &\
                 ( ((t-TwoP_trigger_delay)%framePeriod)<(framePeriod-security) )  ] = 0. # switch to 0
    return 1.0*array

# output_funcs = [Scan1Plane_Screen342V.trigger2P,
#                 off_scan_periods]
output_funcs = [Scan1Plane_Screen342V.trigger2P]

