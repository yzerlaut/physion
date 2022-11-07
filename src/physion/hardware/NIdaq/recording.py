import nidaqmx, time
import numpy as np

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import (
    AnalogSingleChannelReader, AnalogMultiChannelReader, DigitalMultiChannelReader)
from nidaqmx.stream_writers import (
    AnalogSingleChannelWriter, AnalogMultiChannelWriter , DigitalMultiChannelWriter)

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from hardware_control.NIdaq.config import find_x_series_devices, find_m_series_devices, get_analog_input_channels,\
    get_analog_output_channels, get_digital_input_channels

def rec_only(device, t_array, inputs):

    dt = t_array[1]-t_array[0]
    sampling_rate = 1./dt
    
    # if outputs.shape[0]>0:
    input_channels = get_analog_input_channels(device)[:inputs.shape[0]]
        
    with nidaqmx.Task() as read_task,  nidaqmx.Task() as sample_clk_task:

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            '{0}/ctr0'.format(device.name), freq=sampling_rate)
        sample_clk_task.timing.cfg_implicit_timing(
            samps_per_chan=len(t_array))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(device.name)

        read_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(input_channels),
            max_val=10, min_val=-10)
        read_task.timing.cfg_samp_clk_timing(
                sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=len(t_array))

        reader = AnalogMultiChannelReader(read_task.in_stream)

        # Start the read task before starting the sample clock source task.
        read_task.start()
        sample_clk_task.start()
            
        reader.read_many_sample(
            inputs, number_of_samples_per_channel=len(t_array),
            timeout=t_array[-1]+2*dt)

        
def stim_and_rec(device, t_array,
                 analog_inputs, analog_outputs,
                 N_digital_inputs=0):

    dt = t_array[1]-t_array[0]
    sampling_rate = 1./dt
    
    # if analog_outputs.shape[0]>0:
    output_analog_channels = get_analog_output_channels(device)[:analog_outputs.shape[0]]
    input_analog_channels = get_analog_input_channels(device)[:analog_inputs.shape[0]]
    if N_digital_inputs >0:
        input_digital_channels = get_digital_input_channels(device)[:N_digital_inputs]
    
    with nidaqmx.Task() as write_analog_task, nidaqmx.Task() as read_analog_task,\
         nidaqmx.Task() as read_digital_task,  nidaqmx.Task() as sample_clk_task:

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        sample_clk_task.co_channels.add_co_pulse_chan_freq(
            '{0}/ctr0'.format(device.name), freq=sampling_rate)
        sample_clk_task.timing.cfg_implicit_timing(
            samps_per_chan=len(t_array))

        samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(device.name)

        ### ---- OUTPUTS ---- ##
        write_analog_task.ao_channels.add_ao_voltage_chan(
            flatten_channel_string(output_analog_channels),
                                   max_val=10, min_val=-10)
        write_analog_task.timing.cfg_samp_clk_timing(
            sampling_rate, source=samp_clk_terminal,
            active_edge=Edge.RISING, samps_per_chan=len(t_array))

        ### ---- INPUTS ---- ##
        read_analog_task.ai_channels.add_ai_voltage_chan(
            flatten_channel_string(input_analog_channels),
            max_val=10, min_val=-10)
        if N_digital_inputs >0:
            read_digital_task.di_channels.add_di_chan(
                flatten_channel_string(input_digital_channels))
            
        read_analog_task.timing.cfg_samp_clk_timing(
                sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=len(t_array))
        if N_digital_inputs >0:
            read_digital_task.timing.cfg_samp_clk_timing(
                sampling_rate, source=samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=len(t_array))

        analog_writer = AnalogMultiChannelWriter(write_analog_task.out_stream)
        analog_reader = AnalogMultiChannelReader(read_analog_task.in_stream)
        if N_digital_inputs >0:
            digital_reader = DigitalMultiChannelReader(read_digital_task.in_stream)

        analog_writer.write_many_sample(analog_outputs)

        # Start the read and write tasks before starting the sample clock
        # source task.
        read_analog_task.start()
        if N_digital_inputs>0:
            read_digital_task.start()
        write_analog_task.start()
        sample_clk_task.start()
            
        analog_reader.read_many_sample(
            analog_inputs, number_of_samples_per_channel=len(t_array),
            timeout=t_array[-1]+2*dt)
        if N_digital_inputs >0:
            digital_inputs = np.zeros((1,len(t_array)), dtype=np.uint32)
            digital_reader.read_many_sample_port_uint32(
                digital_inputs, number_of_samples_per_channel=len(t_array),
                timeout=t_array[-1]+2*dt)
        else:
            digital_inputs = None
    return analog_inputs, digital_inputs
        
        
if __name__=='__main__':

    import argparse
    # First a nice documentation 
    parser=argparse.ArgumentParser(description="Record data and send signals through a NI daq card",
                                   formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-os', "--output_signal", help="npy file for an array of output signal", default='')
    parser.add_argument('-Nai', "--Nchannel_analog_rec", help="Number of analog input channels to be recorded ", type=int, default=2)
    parser.add_argument('-Ndi', "--Nchannel_digital_rec", help="Number of digital input channels to be recorded ", type=int, default=4)
    parser.add_argument('-dt', "--acq_time_step", help="Temporal sampling (in s): 1/acquisition_frequency ", type=float, default=1e-4)
    parser.add_argument('-T', "--recording_time", help="Length of recording time in (s)", type=float, default=3)
    parser.add_argument('-f', "--filename", help="filename",type=str, default='data.npy')
    parser.add_argument('-d', "--device", help="device name", type=str, default='')
    args = parser.parse_args()

    if args.device=='':
        m_devices, x_devices = find_m_series_devices(), find_x_series_devices()
        if len(m_devices)>0:
            args.device = m_devices[0]
        elif len(x_devices)>0:
            args.device = x_devices[0]

    print(args.device)
    print('Digital input channels: ', get_digital_input_channels(args.device))

    t_array = np.arange(int(args.recording_time/args.acq_time_step))*args.acq_time_step
    analog_inputs = np.zeros((args.Nchannel_analog_rec,len(t_array)))

    analog_outputs = 100*np.array([5e-2*np.sin(2*np.pi*t_array),
                                   2e-2*np.sin(2*np.pi*t_array)])


    
    print('running rec & stim [...]')
    analog_inputs, digital_inputs = stim_and_rec(args.device, t_array, analog_inputs, analog_outputs,
                                                 args.Nchannel_digital_rec)
    # print(digital_inputs)
    # tstart = 1e3*time.time()
    # print('writing T=%.1fs of recording (at f=%.2fkHz, across N=%i channels) in : %.2f ms' % (T, 1e-3/dt,inputs.shape[0],1e3*time.time()-tstart))
    # print('Running 5 rec only')
    # for i in range(5):
    #     tstart = 1e3*time.time()
    #     np.save('data.npy', inputs)
    #     print('writing T=%.1fs of recording (at f=%.2fkHz, across N=%i channels) in : %.2f ms' % (T, 1e-3/dt,inputs.shape[0],1e3*time.time()-tstart))
    # rec_only(args.device, t_array, inputs)
    # np.save(args.filename, analog_inputs)

    import matplotlib.pylab as plt
    fig, AX = plt.subplots(args.Nchannel_analog_rec-1)
    for i, l in zip([1,2], ['A','B']):
        signal = analog_inputs[i]
        # if signal.std()>0:
        #     plt.plot(t_array, i+(signal-signal.min())/(signal.max()-signal.min()))
        # else:
        AX[i-1].plot(t_array, signal)
        AX[i-1].set_ylabel('Channel %s (V)' % l)
        
        
    # plt.plot(1e3*t_array, analog_inputs[0,:])
    plt.xlabel('time (ms)')
    plt.show()
