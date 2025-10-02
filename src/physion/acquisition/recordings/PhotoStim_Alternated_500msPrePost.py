import numpy as np

from . import Scan1Plane_Screen342V

TwoP_trigger_delay = Scan1Plane_Screen342V.TwoP_trigger_delay

def optogen_drive(t, stim, cmd_level):
    array = np.zeros(len(t), dtype=float) # 1 by default
    for i, tstart in enumerate(stim.experiment['time_start']):
        if i%2==1:
            # alternated
            cond = (t> (TwoP_trigger_delay+tstart-0.5) ) &\
                    (t< (TwoP_trigger_delay+stim.experiment['time_stop'][i]+0.5) )
            array[cond] = cmd_level
            print(i, tstart)

    return array

output_funcs = [optogen_drive]

