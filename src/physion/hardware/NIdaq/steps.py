import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from hardware_control.NIdaq.recording import *


if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="Record data and send signals through a NI daq card",
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-Nai', "--Nchannel_analog_rec", type=int, default=1)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/facq", type=float, default=1e-4)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=10)
    parser.add_argument("--on", help="on phase duration  (ms)", type=float, default=2500)
    parser.add_argument("--off", help="off phase duration (ms)", type=float, default=2500)
    parser.add_argument('-p', "--pulse_value", type=float, default=5.)
    parser.add_argument('-f', "--filename", help="filename",type=str, default='data.npy')
    parser.add_argument('-d', "--device", help="device name", type=str, default='')
    args = parser.parse_args()

    if args.device=='':
        try:
            args.device = find_m_series_devices()[0]
        except BaseException as be:
            pass
        try:
            args.device = find_x_series_devices()[0]
        except BaseException as be:
            pass
    print(args.device)
        
    t_array = np.arange(int(args.recording_time/args.acq_time_step))*args.acq_time_step
    analog_inputs = np.zeros((args.Nchannel_analog_rec,len(t_array)))

    array = 0*t_array
    for i in range(int(args.recording_time*1e3/(args.on+args.off))+1):
       cond = (1e3*t_array>i*(args.on+args.off)) & (1e3*t_array<(i*(args.on+args.off)+args.on))
       array[cond] = args.pulse_value
		
    array[0] = 0
    array[-1] = 0
    analog_outputs = np.array([array])
    
    print('running rec & stim [...]')
    analog_inputs, digital_inputs = stim_and_rec(args.device, t_array, analog_inputs, analog_outputs)

    #import matplotlib.pylab as plt
    #for i in range(args.Nchannel_analog_rec):
    #   plt.plot(t_array[::10], analog_inputs[i][::10])
    #plt.show()
