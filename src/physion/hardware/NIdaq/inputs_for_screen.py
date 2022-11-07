import sys, os, pathlib
sys.path.append(os.path.join(os.path.expanduser('~'), 'analyz'))
from analyz.IO.axon_to_python import load_file
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from hardware_control.NIdaq.recording import *

def load_abf(filename, sampling=0.1):

    t, v = load_file(filename)
    
    return np.array(v[0][int(sampling/(t[1]-t[0]))::])


if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-Nai', "--Nchannel_analog_rec", type=int, default=1)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/facq", type=float, default=5e-5)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=2)
    parser.add_argument('-s', "--scale_Pockel_signal", type=float, default=1.)
    parser.add_argument('-p', "--constant_pulse_value", type=float, default=5.)
    parser.add_argument('-d', "--device", help="device name", type=str, default='')
    parser.add_argument('-fn', "--filename", help="device name", type=str, default=os.path.join(os.path.expanduser('~'), 'Downloads', '2021_03_03_0007.abf'))
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

    analog_outputs = np.array([load_abf(args.filename, sampling=args.acq_time_step)[:len(t_array)]*args.scale_Pockel_signal,
                               args.constant_pulse_value+0*t_array])
    #analog_outputs = np.array([args.constant_pulse_value+0*t_array,
    #		       load_abf(args.filename, sampling=args.acq_time_step)[:len(t_array)]*args.scale_Pockel_signal])

    analog_outputs[0,0], analog_outputs[0,-1] = 0, 0
    analog_outputs[1,0], analog_outputs[1,-1] = 0, 0
    
    print('running rec & stim [...]')
    analog_inputs, digital_inputs = stim_and_rec(args.device, t_array,
                                                 analog_inputs, analog_outputs)

    import matplotlib.pylab as plt
    for i in range(args.Nchannel_analog_rec):
      plt.plot(t_array[:3000], analog_inputs[i][:3000])
    plt.show()
