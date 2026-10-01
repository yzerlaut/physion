"""
Interface for (cheap) USB DAQ from NI
    e.g. USB-6002

    (WIP)
"""
import nidaqmx
import time, threading
import numpy as np

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge, WAIT_INFINITELY
from nidaqmx.stream_readers import (
    AnalogMultiChannelReader,
    DigitalMultiChannelReader
)
from nidaqmx.stream_writers import AnalogMultiChannelWriter

from physion.hardware.NIdaq.config import\
         find_usb_devices, get_digital_input_channels,\
         get_analog_input_channels, get_analog_output_channels

class Acquisition:
    """
    the input buffers are read by "reading_task_callback" on the NIdaq thread
        (as in physion.hardware.NIdaq.main):
        - they are stored in a list, concatenated in "close"
            (no copy of the whole recording at each buffer)
        - "self.lock" protects them: "close" waits for a callback in progress
            and the callbacks do nothing after the acquisition is stopped
    """

    def __init__(self,
                 sampling_rate=1000,
                 Nchannel_analog_in=0,
                 Nchannel_digital_in=1,
                 max_time=10,
                 buffer_time=0.5,
                 filename=None,
                 device=None,
                 outputs=None,
                 output_steps=[], # should be a set of dictionaries, output_steps=[{'channel':0, 'onset': 2.3, 'duration': 1., 'value':5}]
                 output_funcs=None, # should be a set of functions func(t) 
                 verbose=False):
        
        self.running, self.data_saved = False, False
        self.lock = threading.Lock()
        self.analog_buffers, self.digital_buffers = [], []

        self.sampling_rate = sampling_rate
        self.dt = 1./self.sampling_rate

        self.buffer_size = int(buffer_time*self.sampling_rate)
        self.Nsamples = int(max_time/buffer_time)*self.buffer_size # ENFORCE multiple of buffer time !! 
        self.max_time = self.Nsamples/self.sampling_rate
        self.Nchannel_analog_in = Nchannel_analog_in
        self.Nchannel_digital_in = Nchannel_digital_in
        self.filename = filename

        if device is None:
            self.device = find_usb_devices()[0]
        else:
            self.device = device

        # preparing input channels
        # - analog:
        self.analog_data = np.zeros((Nchannel_analog_in, 0), dtype=np.float64)
        if self.Nchannel_analog_in>0:
            self.analog_input_channels = \
                    get_analog_input_channels(self.device)[:Nchannel_analog_in]
        # - digital:
        self.digital_data = np.zeros((1, 0), dtype=np.uint32)
        if self.Nchannel_digital_in>0:
            self.digital_input_channels = \
                    get_digital_input_channels(self.device)[:Nchannel_digital_in]

        # preparing output channels
        if outputs is not None: # used as a flag for output or not
            # -
            self.output_channels = \
                    get_analog_output_channels(self.device)[:outputs.shape[0]]

        elif output_funcs is not None:
            # -
            Nchannel = len(output_funcs)
            self.output_channels = \
                    get_analog_output_channels(self.device)[:Nchannel]
            t = np.arange(int(self.Nsamples))*self.dt
            outputs = np.zeros((Nchannel,len(t)))
            for i, func in enumerate(output_funcs):
                outputs[i] = func(t)

        elif len(output_steps)>0:
            # -
            Nchannel = max([d['channel'] for d in output_steps])+1
            # have to be elements 
            t = np.arange(int(self.Nsamples))*self.dt
            outputs = np.zeros((Nchannel,len(t)))
            # add as many channels as necessary
            for step in output_steps:
                if step['channel']>outputs.shape[0]:
                    outputs =  np.append(outputs, np.zeros((1,len(t))), axis=0)
            for step in output_steps:
                cond = (t>step['onset']) & (t<=step['onset']+step['duration'])
                outputs[step['channel']][cond] = step['value']
            self.output_channels = get_analog_output_channels(self.device)[:outputs.shape[0]]


        self.outputs = outputs      


            
    def launch(self):

        if self.outputs is not None:
            self.write_task = nidaqmx.Task()

        if self.Nchannel_analog_in>0:
            self.read_analog_task = nidaqmx.Task()
        if self.Nchannel_digital_in>0:
            self.read_digital_task = nidaqmx.Task()
        self.sample_clk_task = nidaqmx.Task()

        # Use a counter output pulse train task as the sample clock source
        # for both the AI and AO tasks.
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq('{0}/ctr0'.format(self.device.name),
                                                                freq=self.sampling_rate)
        self.sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=int(self.Nsamples))
        self.samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(self.device.name)

        ### ---- OUTPUTS ---- ##
        if self.outputs is not None:
            self.write_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(self.output_channels),
                max_val=10, min_val=-10)
            self.write_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, 
                samps_per_chan=int(self.Nsamples))
        
        ### ---- INPUTS ---- ##
        if self.Nchannel_analog_in>0:
            self.read_analog_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(self.analog_input_channels),
                max_val=10, min_val=-10)
            
        if self.Nchannel_digital_in>0:
            self.read_digital_task.di_channels.add_di_chan(
                flatten_channel_string(self.digital_input_channels))

        if self.Nchannel_analog_in>0:
            self.read_analog_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples))
            
        if self.Nchannel_digital_in>0:
            self.read_digital_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples))
        
        if self.Nchannel_analog_in>0:
            self.analog_reader = AnalogMultiChannelReader(self.read_analog_task.in_stream)
        if self.Nchannel_digital_in>0:
            self.digital_reader = DigitalMultiChannelReader(self.read_digital_task.in_stream)

        # callback on ONE task only (it reads both): avoids double-reading
        primary = self.read_analog_task if self.Nchannel_analog_in>0 else\
                    (self.read_digital_task if self.Nchannel_digital_in>0 else None)
        if primary is not None:
            primary.register_every_n_samples_acquired_into_buffer_event(self.buffer_size,
                                                                        self.reading_task_callback)

        if self.outputs is not None:
            self.writer = AnalogMultiChannelWriter(self.write_task.out_stream)
            self.writer.write_many_sample(self.outputs)

        self.analog_buffers, self.digital_buffers = [], []
        self.running, self.data_saved = True, False

        # Start the read task before starting the sample clock source task.            
        if self.Nchannel_analog_in>0:
            self.read_analog_task.start() 
        if self.Nchannel_digital_in>0:
            self.read_digital_task.start()
        
        if self.outputs is not None:
            self.write_task.start()
            
        self.sample_clk_task.start()

        if self.filename is not None:
            self.t0 = time.time()
            # saving the time stamp of the start !
            np.save(self.filename.replace('.npy', '.start.npy'), 
                    self.t0*np.ones(1))
        
    def close(self, return_data=False):
        """
        not optimal...
        
        nidaqmx has weird behaviors sometimes... :(
        """
        # stop the reading callbacks before closing the tasks:
        #   waits for a callback in progress (lock), the next ones do nothing
        with self.lock:
            was_running, self.running = self.running, False

        if was_running:
            if self.Nchannel_digital_in>0:
                self.read_digital_task.close()
            if self.Nchannel_analog_in>0:
                self.read_analog_task.close()
            if self.outputs is not None:
                self.write_task.close()
            self.sample_clk_task.close()

        if not self.data_saved:
            self.gather_data()

        if (self.filename is not None):
            if self.data_saved:
                print('[ok] NIdaq data already saved as: %s ' % self.filename)
            else:
                np.save(self.filename,
                        {'analog':self.analog_data,
                         'digital':self.digital_data,
                         'dt':self.dt})
                print('[ok] NIdaq data saved as: %s ' % self.filename)
            self.data_saved = True

        if return_data:
            return self.analog_data, self.digital_data, self.dt

    def gather_data(self):
        """
        concatenates the buffers read during the acquisition
            into self.analog_data (channels, samples)
             and self.digital_data (1, samples)
        both cut to the same number of samples
        """
        if len(self.analog_buffers)>0:
            self.analog_data = np.concatenate([self.analog_data]+self.analog_buffers, axis=1)
        if len(self.digital_buffers)>0:
            self.digital_data = np.concatenate([self.digital_data]+self.digital_buffers, axis=1)
        self.analog_buffers, self.digital_buffers = [], []

        if (self.Nchannel_analog_in>0) and (self.Nchannel_digital_in>0) and\
                (self.analog_data.shape[1]!=self.digital_data.shape[1]):
            n = min([self.analog_data.shape[1], self.digital_data.shape[1]])
            print(' [!!] NIdaq: %i analog and %i digital samples, cut to %i ' %\
                    (self.analog_data.shape[1], self.digital_data.shape[1], n))
            self.analog_data = self.analog_data[:,:n]
            self.digital_data = self.digital_data[:,:n]

    def reading_task_callback(self, task_idx, event_type, num_samples, callback_data=None):
        """
        called on the NIdaq thread every "buffer_size" samples
            (stored in buffers, see "gather_data")
        """
        with self.lock:

            if not self.running:
                # acquisition stopped ("close"), the tasks are being closed
                return 0

            try:
                if self.Nchannel_analog_in>0:
                    analog_buffer = np.zeros((self.Nchannel_analog_in, num_samples), dtype=np.float64)
                    self.analog_reader.read_many_sample(analog_buffer, num_samples, timeout=WAIT_INFINITELY)

                if self.Nchannel_digital_in>0:
                    digital_buffer = np.zeros((1, num_samples), dtype=np.uint32)
                    self.digital_reader.read_many_sample_port_uint32(digital_buffer,
                                                                 num_samples, timeout=WAIT_INFINITELY)

                # both stored together: same number of analog and digital samples
                if self.Nchannel_analog_in>0:
                    self.analog_buffers.append(analog_buffer)
                if self.Nchannel_digital_in>0:
                    self.digital_buffers.append(digital_buffer)

            except nidaqmx.errors.DaqError as error:
                print(' [!!] NIdaq: buffer of %i samples not read ' % num_samples, error)

        return 0  # needed for this callback to be well defined (see nidaqmx doc).


    def select_device(self):
        success = False
        try:
            self.device = find_x_series_devices()[0]
            print('[ok] X-series card found:', self.device)
            success = True
        except BaseException: 
            # print('no X-series card found')
            pass
        try:
            self.device = find_m_series_devices()[0]
            print('[ok] M-series card found:', self.device)
            success = True
        except BaseException:
            # print('no M-series card found')
            pass

        if not success:
            print('Neither M-series nor X-series NI DAQ card found')

        
if __name__=='__main__':

    # Simple Test: Connect together AO0 and AI0 
    # --> we send a pulse in AO0
    acq = Acquisition(sampling_rate=1000,
                      Nchannel_analog_in=1,
                      Nchannel_digital_in=2,
                      output_steps=[{'channel':0, 'onset': 1., 'duration': 1., 'value':5}],
                      filename='data.npy')
    acq.launch()
    tstart = time.time()
    while (time.time()-tstart)<3.:
        pass
    acq.close()
    # --> should appear in AI0:
    import matplotlib.pylab as plt
    plt.plot(acq.analog_data[0])
    plt.show()
    
    # print(acq.digital_data.shape)
    # np.save('data.npy', acq.analog_data)
    # from datavyz import ge
    # ge.plot(acq.digital_data[1,:][::10])
    # ge.show()
