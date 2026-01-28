"""
Interface for M-Series and X-Series NIdaq cards

"""

import time
import numpy as np
import nidaqmx

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge, WAIT_INFINITELY, LineGrouping
from nidaqmx.stream_readers import (
    AnalogMultiChannelReader,
    DigitalMultiChannelReader
)
from nidaqmx.stream_writers import (
    AnalogMultiChannelWriter,
    DigitalMultiChannelWriter
)

from physion.hardware.NIdaq.config import (
    find_x_series_devices, find_m_series_devices,
    get_analog_input_channels, get_analog_output_channels
)

class Acquisition:
    """

    sets up hardware-timed clock on the analog and digital output channels

    by default, you can initialize outputs (digital and analog) by passing
    either a set of step dictionaries, or a set of functions (of time)

    DIGITAL data: np.uint32
    ANALOG data: np.float64

    for digital channels, **only "port0" supports buffered data** 
        --> only this can be used with this interface !!
    PUT the DIGITAL OUTPUTS on the first lines, e.g. line0-3
    and the DIGITAL INPUTS on the remaining ones, e.g. line4-7
    
    Usage:

        acq = Acquisition(...)

        # if needed, here you can modify the output values after initialization
        acq.analog_outputs = ...
        acq.digital_outputs = ...

        # launch the acquisition
        acq.launch()    

        # wait that acquisition finishes...
        while acq.running:
            time.sleep(1)

        # here you 
        print(acq.analog_data)
        print(acq.digital_data)

        # close the process and free up memory
        acq.close()

    """

    def __init__(self,
                 sampling_rate=10000,
                 max_time=10,
                 buffer_time=0.5,
                 filename=None,
                 device=None,

                 # ---- Analog inputs ---- #
                 Nchannel_analog_in=0,

                 # ---- Analog outputs ----- #
                 analog_outputs=None,
                 analog_output_funcs=None,      # [func(t)] -> waveform (-10, 10)V range !!
                 analog_output_steps=[],        # [{'channel':0,'onset':2.3,'duration':1.,'value':5}]

                 # ---- Digital inputs ---- #
                 digital_input_port="port0/line0:1",    # set to None to disable DI

                 # ---- Digital output ---- #
                 digital_output_port="port0/line2:3",   # set None to disable digital out
                 digital_output_funcs=None,     # [func(t)] -> waveform (0, 1) binary func
                 digital_output_steps=[],       # [{'channel':0,'onset':2.3,'duration':1.}]

                 verbose=False):

        self.verbose = verbose
        self.running, self.data_saved = False, False

        self.sampling_rate = float(sampling_rate)
        self.dt = 1.0 / self.sampling_rate
        self.buffer_size = int(buffer_time * self.sampling_rate)

        # enforce Nsamples as multiple of buffer_size
        nbuf = int(max_time / buffer_time)
        self.Nsamples = nbuf * self.buffer_size
        self.max_time = self.Nsamples / self.sampling_rate  # FIX: seconds

        self.Nchannel_analog_in = int(Nchannel_analog_in)
        self.filename = filename

        # device selection
        if device is None:
            self.select_device()
        else:
            self.device = device

        devname = self.device.name  # e.g. "Dev1"

        # ---- Analog IN channels ----
        if self.Nchannel_analog_in > 0:
            self.analog_data = np.zeros((self.Nchannel_analog_in, 1), dtype=np.float64)
            self.analog_input_channels = get_analog_input_channels(self.device)[:self.Nchannel_analog_in]
        else:
            self.analog_data = None

        # ---- Digital IN channels ---- #
        if digital_input_port is not None:
            self.digital_in_chan = f"{devname}/{digital_input_port}"  # e.g. "Dev1/line0:7"
            self.digital_data = np.zeros((1, 1), dtype=np.uint32)  # store as 0/1 in 8 channels, using 8 bit encoding

        else:
            self.digital_data = None

        # ---- Analog OUT channels ---- #
        analog_output_channels = get_analog_output_channels(self.device)

        if analog_output_funcs is not None:

            if len(analog_output_funcs)>len(analog_output_channels):
                print(' too many functions for the number of available analog output',
                      '(n=%i)' % len(self.analog_output_channels))
                print('    ----> ignoring the additional channels ! ')

            Nch = min([len(analog_output_funcs), len(analog_output_channels)])
            t = np.arange(self.Nsamples) * self.dt
            out = np.zeros((Nch, len(t)), dtype=np.float64)
            for i, func in enumerate(analog_output_funcs[:Nch]):
                out[i] = func(t)
            self.output_channels = analog_output_channels[:Nch] 
            self.analog_outputs = out

        elif len(analog_output_steps) > 0:

            Nch = max([d['channel'] for d in analog_output_steps]) + 1

            if Nch>len(analog_output_channels):
                print(' too many step channels for the number of available analog output',
                      '(n=%i)' % len(analog_output_channels))
                print('    ----> ignoring the additional channels ! ')
                Nch = len(analog_output_channels) 

            t = np.arange(self.Nsamples) * self.dt
            out = np.zeros((Nch, len(t)), dtype=np.float64)
            for step in analog_output_steps:
                cond = (t > step['onset']) & (t <= step['onset'] + step['duration'])
                out[step['channel']][cond] = step['value']
            self.output_channels = analog_output_channels[:Nch] 
            self.analog_outputs = out

        else:
            self.analog_outputs = None

        # ---- Digital OUT (binary port waveform) ----

        if digital_output_port is not None:

            self.digital_out_chan = f"{devname}/{digital_output_port}"  # e.g. "Dev1/port0/line4:6"

            if digital_output_funcs is not None:

                Nch = len(digital_output_funcs)
                t = np.arange(self.Nsamples) * self.dt
                out = np.zeros((1, len(t)), dtype=np.uint32)
                for i, func in enumerate(digital_output_funcs[:Nch]):
                    out[0,:] += func(t)*(2**i) # func should be [0,1] uint32 output
                self.digital_outputs = out

            elif len(digital_output_steps) > 0:

                Nch = max([d['channel'] for d in digital_output_steps]) + 1

                t = np.arange(self.Nsamples) * self.dt
                out = np.zeros((1, len(t)), dtype=np.uint32)
                for step in digital_output_steps:
                    cond = (t > step['onset']) & (t <= step['onset'] + step['duration'])
                    out[0,cond] += 2**step['channel']
                self.digital_outputs = out

            else:
                self.digital_outputs = None

        else:
            self.digital_outputs = None


    def launch(self):
        devname = self.device.name

        # Tasks
        self.sample_clk_task = nidaqmx.Task('clock')

        self.read_analog_task = nidaqmx.Task('analog-read')\
                                 if self.Nchannel_analog_in > 0 else None
        self.read_digital_task = nidaqmx.Task('digital-read')\
                                 if self.digital_in_chan is not None else None

        self.write_analog_task = nidaqmx.Task('analog-write')\
                                 if self.analog_outputs is not None else None
        self.write_digital_task = nidaqmx.Task('digital-write')\
                                 if self.digital_outputs is not None else None

        # --- Sample clock from counter (ctr0) ---
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq(
            f"{devname}/ctr0", freq=self.sampling_rate
        )
        self.sample_clk_task.timing.cfg_implicit_timing(\
                                        samps_per_chan=int(self.Nsamples))
        self.samp_clk_terminal = f"/{devname}/Ctr0InternalOutput"

        # --- ANALOG OUTPUTS ---
        if self.write_analog_task is not None:
            self.write_analog_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(self.output_channels),
                max_val=10, min_val=-10
            )
            self.write_analog_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, 
                source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, 
                samps_per_chan=int(self.Nsamples)
            )

        # --- DIGITAL OUTPUTS ---
        if self.write_digital_task is not None:
            # Binary port output: one channel = port0, 32 lines packed into uint32
            self.write_digital_task.do_channels.add_do_chan(
                self.digital_out_chan,
                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )
            self.write_digital_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, 
                source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, 
                samps_per_chan=int(self.Nsamples)
            )

        # --- ANALOG INPUTS ---
        if self.read_analog_task is not None:
            self.read_analog_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(self.analog_input_channels),
                max_val=10, min_val=-10
            )
            self.read_analog_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, 
                source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, 
                samps_per_chan=int(self.Nsamples)
            )

        # --- DIGITAL INPUTS ---
        if self.read_digital_task is not None:
            # Read single TTL line (PFI0) as DI
            self.read_digital_task.di_channels.add_di_chan(\
                                                self.digital_in_chan)
            self.read_digital_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, 
                samps_per_chan=int(self.Nsamples)
            )

        # Readers
        self.analog_reader = None
        if self.read_analog_task is not None:
            self.analog_reader = AnalogMultiChannelReader(self.read_analog_task.in_stream)
            # self.read_analog_task.register_every_n_samples_acquired_into_buffer_event(self.buffer_size,
                                                                                    #   self.reading_task_callback)

        self.digital_reader = None
        if self.read_digital_task is not None:
            self.digital_reader = DigitalMultiChannelReader(self.read_digital_task.in_stream)
            # self.read_digital_task.register_every_n_samples_acquired_into_buffer_event(self.buffer_size,
                                                                                    #    self.reading_task_callback)

        # Writers (preload buffers before starting sample clock)
        if self.write_analog_task is not None:
            self.analog_writer = AnalogMultiChannelWriter(self.write_analog_task.out_stream, 
                                                          auto_start=False)
            self.analog_writer.write_many_sample(self.analog_outputs)

        if self.write_digital_task is not None:
            self.digital_writer = DigitalMultiChannelWriter(self.write_digital_task.out_stream, 
                                                            auto_start=False)
            # shape (1, N) uint32
            self.digital_writer.write_many_sample_port_uint32(self.digital_outputs)

        # Register callback ONLY ON ONE PRIMARY TASK (avoids double-reading)
        primary = None
        if self.read_analog_task is not None:
            primary = self.read_analog_task
        elif self.read_digital_task is not None:
            primary = self.read_digital_task

        if primary is not None:
            primary.register_every_n_samples_acquired_into_buffer_event(
                self.buffer_size, self.reading_task_callback
            )

        # Start input tasks first
        if self.read_analog_task is not None:
            self.read_analog_task.start()
        if self.read_digital_task is not None:
            self.read_digital_task.start()

        # Start output tasks (they wait on sample clock)
        if self.write_analog_task is not None:
            self.write_analog_task.start()
        if self.write_digital_task is not None:
            self.write_digital_task.start()

        # Start the sample clock last
        self.sample_clk_task.start()

        if self.filename is not None:
            self.t0 = time.time()
            np.save(self.filename.replace('.npy', '.start.npy'), self.t0 * np.ones(1))

        self.running, self.data_saved = True, False

    def reformat_digital_data(self):
        """"
        converts back the uint32 encoding of multiple lines into a set of booleans
        """
        n0 = int(self.digital_in_chan.split('line')[1].split(':')[0])
        n1 = int(self.digital_in_chan.split(':')[1])
        digital_rf = np.zeros((n1-n0+1, self.digital_data.shape[1]), 
                               dtype=bool)
        for i, n in enumerate(range(n0, n1+1)):
            digital_rf[i,:] = np.bool_(\
                 (self.digital_data[0,:] >> n) % 2 )
        self.digital_data = digital_rf

    def close(self):

        if self.running:
            for task in [self.read_digital_task, self.read_analog_task,
                         self.write_digital_task, self.write_analog_task,
                         self.sample_clk_task]:
                if task is not None:
                    task.close()

        if self.filename is not None:

            if not self.data_saved:

                if self.digital_data is not None:
                    self.reformat_digital_data()
                    digital = self.digital_data[:,1:]
                else:
                    digital = None

                np.save(self.filename, {
                    'analog': self.analog_data[:, 1:] if self.analog_data is not None else None,
                    'digital': digital,
                    'dt': self.dt
                })
                print('[ok] NIdaq data saved as:', self.filename)

            self.data_saved = True

        self.running = False


    def reading_task_callback(self, task_idx, event_type, num_samples, callback_data=None):

        if not self.running:
            self.close()
            return 0

        try:
            if self.read_analog_task is not None:
                analog_buffer = np.zeros((self.Nchannel_analog_in, num_samples), dtype=np.float64)
                self.analog_reader.read_many_sample(analog_buffer, num_samples, 
                                                    timeout=WAIT_INFINITELY)
                self.analog_data = np.append(self.analog_data, analog_buffer, axis=1)

            if self.read_digital_task is not None:
                digital_buffer = np.zeros((1,num_samples), dtype=np.uint32)
                self.digital_reader.read_many_sample_port_uint32(digital_buffer, 
                                                                 num_samples, 
                                                                 timeout=WAIT_INFINITELY)
                dig_u8 = digital_buffer.astype(np.uint8).reshape(1, -1)
                self.digital_data = np.append(self.digital_data, dig_u8, 
                                              axis=1)

        except nidaqmx.errors.DaqError:
            pass

        return 0

    def select_device(self):
        success = False
        try:
            self.device = find_x_series_devices()[0]
            print('[ok] X-series card found:', self.device)
            success = True
        except BaseException:
            pass
        try:
            self.device = find_m_series_devices()[0]
            print('[ok] M-series card found:', self.device)
            success = True
        except BaseException:
            pass

        if not success:
            raise RuntimeError("Neither M-series nor X-series NI DAQ card found")


if __name__ == '__main__':

    tstop = 3
    acq = Acquisition(
        sampling_rate=1000,
        Nchannel_analog_in=1,
        max_time=tstop,
        digital_output_port="port0/line0:1",
        digital_input_port="port0/line3:4",
        digital_output_steps=[
            {'channel':0, 'onset':0.1, 'duration':0.1},
            {'channel':1, 'onset':0.5, 'duration':0.1},
            {'channel':0, 'onset':1.1, 'duration':0.5},
            {'channel':0, 'onset':2.1, 'duration':0.5},
        ],
        filename='data.npy',
        verbose=True
    )
    acq.launch()
    t0 = time.time()
    while (time.time()-t0)<tstop:
        time.sleep(0.2)
    acq.close()
    
    print(acq.digital_data)
    print(acq.analog_data)
    # --> should appear in AI0:
    import matplotlib.pylab as plt
    fig, ax = plt.subplots(1, figsize=(8,4))
    plt.plot(acq.analog_data[0,:])
    fig, ax = plt.subplots(1, figsize=(8,4))
    plt.plot(acq.digital_data[0])
    plt.show()
    
    # print(acq.digital_data.shape)
    # np.save('data.npy', acq.analog_data)
    # from datavyz import ge
    # ge.plot(acq.digital_data[1,:][::10])
    # ge.show()
