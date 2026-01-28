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
    DigitalSingleChannelReader
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
    Adds hardware-timed DIGITAL OUTPUT on port0 (P0.0..P0.7) using DigitalMultiChannelWriter.

    - Digital OUT (binary):  DevX/port0   (uint8 samples, bits map to P0.0..P0.7)
    - Digital IN (loopback): DevX/PFI0    (bool samples)
    """

    def __init__(self,
                 sampling_rate=10000,
                 Nchannel_analog_in=2,
                 max_time=10,
                 buffer_time=0.5,
                 filename=None,
                 device=None,

                 # ---- Analog outputs (existing behavior) ----
                 outputs=None,
                 output_steps=[],      # [{'channel':0,'onset':2.3,'duration':1.,'value':5}]
                 output_funcs=None,    # func(t) -> waveform

                 # ---- NEW: Digital output (binary port0) ----
                 digital_out_port="port0",      # set None to disable digital out
                 digital_out_waveform=None,     # shape (1, Nsamples) uint8, each sample is 0..255 for port bits
                 sync_bit=0,                    # bit index for sync TTL (P0.0 default)
                 sync_hz=5,                     # default sync frequency if waveform not provided
                 start_bit=1,                   # optional start marker bit (P0.1 default)
                 start_time=0.2,                # seconds
                 start_width=0.01,              # seconds

                 # ---- NEW: Digital input line (for loopback recording) ----
                 digital_in_line="PFI0",        # set None to disable DI
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
        self.analog_data = np.zeros((self.Nchannel_analog_in, 1), dtype=np.float64)
        self.analog_input_channels = []
        if self.Nchannel_analog_in > 0:
            self.analog_input_channels = get_analog_input_channels(self.device)[:self.Nchannel_analog_in]

        # ---- Digital IN (loopback) ----
        self.digital_in_line = digital_in_line
        self.digital_data = np.zeros((1, 1), dtype=np.uint8)  # store as 0/1
        self.digital_in_chan = None
        if self.digital_in_line is not None:
            self.digital_in_chan = f"{devname}/{self.digital_in_line}"  # e.g. "Dev1/PFI0"

        # ---- Analog OUT (existing behavior) ----
        self.outputs = None
        self.output_channels = None

        if outputs is not None:
            self.outputs = outputs
            self.output_channels = get_analog_output_channels(self.device)[:outputs.shape[0]]

        elif output_funcs is not None:
            Nch = len(output_funcs)
            self.output_channels = get_analog_output_channels(self.device)[:Nch]
            t = np.arange(self.Nsamples) * self.dt
            out = np.zeros((Nch, len(t)), dtype=np.float64)
            for i, func in enumerate(output_funcs):
                out[i] = func(t)
            self.outputs = out

        elif len(output_steps) > 0:
            Nch = max([d['channel'] for d in output_steps]) + 1
            t = np.arange(self.Nsamples) * self.dt
            out = np.zeros((Nch, len(t)), dtype=np.float64)
            for step in output_steps:
                cond = (t > step['onset']) & (t <= step['onset'] + step['duration'])
                out[step['channel']][cond] = step['value']
            self.output_channels = get_analog_output_channels(self.device)[:out.shape[0]]
            self.outputs = out

        # ---- Digital OUT (binary port waveform) ----
        self.digital_out_port = digital_out_port
        self.digital_out_chan = None
        self.digital_out = None

        if self.digital_out_port is not None:
            self.digital_out_chan = f"{devname}/{self.digital_out_port}"  # e.g. "Dev1/port0"

            if digital_out_waveform is None:
                self.digital_out = self._build_default_port_waveform(
                    sync_bit=sync_bit,
                    sync_hz=sync_hz,
                    start_bit=start_bit,
                    start_time=start_time,
                    start_width=start_width
                )
            else:
                arr = np.asarray(digital_out_waveform, dtype=np.uint8)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                if arr.shape[0] != 1 or arr.shape[1] != self.Nsamples:
                    raise ValueError(f"digital_out_waveform must be shape (1, Nsamples={self.Nsamples}) uint8")
                self.digital_out = arr

    def _build_default_port_waveform(self, sync_bit=0, sync_hz=5, start_bit=1, start_time=0.2, start_width=0.01):
        """
        Returns shape (1, Nsamples) uint8. Each sample is 0..255 representing port bits.
        bit0 -> P0.0, bit1 -> P0.1, ...
        """
        N = self.Nsamples
        fs = self.sampling_rate
        t = np.arange(N) / fs

        port = np.zeros(N, dtype=np.uint8)

        # Sync square wave on sync_bit
        if sync_hz is not None and sync_hz > 0:
            period = 1.0 / sync_hz
            phase = np.mod(t, period)
            high = (phase < (period / 2.0))
            port[high] |= (1 << int(sync_bit))

        # Start marker pulse on start_bit
        if start_bit is not None:
            s0 = int(start_time * fs)
            s1 = min(N, s0 + int(start_width * fs))
            if s0 < N:
                port[s0:s1] |= (1 << int(start_bit))

        return port.reshape(1, -1)

    def launch(self):
        devname = self.device.name

        # Tasks
        self.sample_clk_task = nidaqmx.Task()

        self.read_analog_task = nidaqmx.Task() if self.Nchannel_analog_in > 0 else None
        self.read_digital_task = nidaqmx.Task() if self.digital_in_chan is not None else None

        self.write_analog_task = nidaqmx.Task() if self.outputs is not None else None
        self.write_digital_task = nidaqmx.Task() if self.digital_out is not None else None

        # --- Sample clock from counter (ctr0) ---
        self.sample_clk_task.co_channels.add_co_pulse_chan_freq(
            f"{devname}/ctr0", freq=self.sampling_rate
        )
        self.sample_clk_task.timing.cfg_implicit_timing(samps_per_chan=int(self.Nsamples))
        self.samp_clk_terminal = f"/{devname}/Ctr0InternalOutput"

        # --- OUTPUTS ---
        if self.write_analog_task is not None:
            self.write_analog_task.ao_channels.add_ao_voltage_chan(
                flatten_channel_string(self.output_channels),
                max_val=10, min_val=-10
            )
            self.write_analog_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples)
            )

        if self.write_digital_task is not None:
            # Binary port output: one channel = port0, 8 lines packed into uint8
            self.write_digital_task.do_channels.add_do_chan(
                self.digital_out_chan,
                line_grouping=LineGrouping.CHAN_FOR_ALL_LINES
            )
            self.write_digital_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples)
            )

        # --- INPUTS ---
        if self.read_analog_task is not None:
            self.read_analog_task.ai_channels.add_ai_voltage_chan(
                flatten_channel_string(self.analog_input_channels),
                max_val=10, min_val=-10
            )
            self.read_analog_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples)
            )

        if self.read_digital_task is not None:
            # Read single TTL line (PFI0) as DI
            self.read_digital_task.di_channels.add_di_chan(self.digital_in_chan)
            self.read_digital_task.timing.cfg_samp_clk_timing(
                self.sampling_rate, source=self.samp_clk_terminal,
                active_edge=Edge.FALLING, samps_per_chan=int(self.Nsamples)
            )

        # Readers
        self.analog_reader = None
        if self.read_analog_task is not None:
            self.analog_reader = AnalogMultiChannelReader(self.read_analog_task.in_stream)

        self.digital_reader = None
        if self.read_digital_task is not None:
            self.digital_reader = DigitalSingleChannelReader(self.read_digital_task.in_stream)

        # Writers (preload buffers before starting sample clock)
        if self.write_analog_task is not None:
            self.analog_writer = AnalogMultiChannelWriter(self.write_analog_task.out_stream, auto_start=False)
            self.analog_writer.write_many_sample(self.outputs)

        if self.write_digital_task is not None:
            self.digital_writer = DigitalMultiChannelWriter(self.write_digital_task.out_stream, auto_start=False)
            # shape (1, N) uint8
            self.digital_writer.write_many_sample_port_uint8(self.digital_out)

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

    def close(self, return_data=False):
        if self.running:
            for task in [self.read_digital_task, self.read_analog_task,
                         self.write_digital_task, self.write_analog_task,
                         self.sample_clk_task]:
                if task is not None:
                    task.close()

        if self.filename is not None:
            if not self.data_saved:
                np.save(self.filename, {
                    'analog': self.analog_data[:, 1:],
                    'digital_in': self.digital_data[:, 1:],   # 0/1 from PFI0
                    'dt': self.dt
                })
                print('[ok] NIdaq data saved as:', self.filename)
            self.data_saved = True

        self.running = False

        if return_data:
            return self.analog_data[:, 1:], self.digital_data[:, 1:], self.dt

    def reading_task_callback(self, task_idx, event_type, num_samples, callback_data=None):
        if not self.running:
            self.close()
            return 0

        try:
            if self.read_analog_task is not None:
                analog_buffer = np.zeros((self.Nchannel_analog_in, num_samples), dtype=np.float64)
                self.analog_reader.read_many_sample(analog_buffer, num_samples, timeout=WAIT_INFINITELY)
                self.analog_data = np.append(self.analog_data, analog_buffer, axis=1)

            if self.read_digital_task is not None:
                dig = np.zeros(num_samples, dtype=np.bool_)
                self.digital_reader.read_many_sample_bool(dig, num_samples, timeout=WAIT_INFINITELY)
                dig_u8 = dig.astype(np.uint8).reshape(1, -1)
                self.digital_data = np.append(self.digital_data, dig_u8, axis=1)

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
    # Example:
    # - Output sync on P0.0 (bit0) as 5 Hz square
    # - Output start pulse on P0.1 (bit1)
    # - Record loopback on PFI0 as digital_in
    # - Record analog on AI0 (e.g. photodiode) if configured by physion config
    acq = Acquisition(
        sampling_rate=1000,
        Nchannel_analog_in=1,
        max_time=3,
        digital_out_port="port0",
        digital_in_line="PFI0",
        sync_bit=0,
        sync_hz=5,
        start_bit=1,
        start_time=0.2,
        start_width=0.01,
        filename='data.npy'
    )
    acq.launch()
    time.sleep(3.1)
    acq.close()
    
    


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
