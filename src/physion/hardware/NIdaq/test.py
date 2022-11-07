import nidaqmx, time, collections, random
import numpy as np

from nidaqmx.constants import Edge, ProductCategory, UsageTypeAI
from nidaqmx.utils import flatten_channel_string
from nidaqmx.stream_readers import (
    AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (
    AnalogSingleChannelWriter, AnalogMultiChannelWriter)

acq_freq = 1000. # seconds
T = 5


def get_analog_loopback_channels(device):
    loopback_channel_pairs = []
    ChannelPair = collections.namedtuple(
        'ChannelPair', ['output_channel', 'input_channel'])

    for ao_physical_chan in device.ao_physical_chans:
        device_name, ao_channel_name = ao_physical_chan.name.split('/')

        loopback_channel_pairs.append(
            ChannelPair(
                ao_physical_chan.name,
                '{0}/_{1}_vs_aognd'.format(device_name, ao_channel_name)
            ))

    return loopback_channel_pairs


def find_x_series_device():
    system = nidaqmx.system.System.local()

    for device in system.devices:
        if (not device.dev_is_simulated and
                device.product_category == ProductCategory.X_SERIES_DAQ and
                len(device.ao_physical_chans) >= 2 and
                len(device.ai_physical_chans) >= 4 and
                len(device.do_lines) >= 8 and
                (len(device.di_lines) == len(device.do_lines)) and
                len(device.ci_physical_chans) >= 4):
            return device


number_of_samples = int(T*acq_freq)
sample_rate = acq_freq # random.uniform(1000, 5000)

t= np.arange(number_of_samples)/acq_freq
waveform  = 0.01*np.sin(2*np.pi*t)

x_series_device = find_x_series_device()

# Select a random loopback channel pair on the device.
loopback_channel_pairs = get_analog_loopback_channels(x_series_device)

number_of_channels = random.randint(2, len(loopback_channel_pairs))
channels_to_test = random.sample(
    loopback_channel_pairs, number_of_channels)

print(channels_to_test)
        
with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task,  nidaqmx.Task() as sample_clk_task:
    
    # Use a counter output pulse train task as the sample clock source
    # for both the AI and AO tasks.
    sample_clk_task.co_channels.add_co_pulse_chan_freq(
        '{0}/ctr0'.format(x_series_device.name), freq=sample_rate)
    sample_clk_task.timing.cfg_implicit_timing(
        samps_per_chan=number_of_samples)

    samp_clk_terminal = '/{0}/Ctr0InternalOutput'.format(
        x_series_device.name)

    write_task.ao_channels.add_ao_voltage_chan(
        flatten_channel_string(
            [c.output_channel for c in channels_to_test]),
        max_val=10, min_val=-10)
    write_task.timing.cfg_samp_clk_timing(
        sample_rate, source=samp_clk_terminal,
        active_edge=Edge.RISING, samps_per_chan=number_of_samples)

    read_task.ai_channels.add_ai_voltage_chan(
        flatten_channel_string(
            [c.input_channel for c in channels_to_test]),
        max_val=10, min_val=-10)
    read_task.timing.cfg_samp_clk_timing(
        sample_rate, source=samp_clk_terminal,
        active_edge=Edge.FALLING, samps_per_chan=number_of_samples)

    writer = AnalogMultiChannelWriter(write_task.out_stream)
    reader = AnalogMultiChannelReader(read_task.in_stream)

    values_to_test = np.array(
        [waveform for _ in range(number_of_channels)], dtype=np.float64)
    writer.write_many_sample(values_to_test)

    # Start the read and write tasks before starting the sample clock
    # source task.
    read_task.start()
    write_task.start()
    sample_clk_task.start()

    values_read = np.zeros(
        (number_of_channels, number_of_samples), dtype=np.float64)
    
    reader.read_many_sample(
        values_read, number_of_samples_per_channel=number_of_samples,
        timeout=5)

    import matplotlib.pylab as plt
    plt.plot(t, waveform)
    plt.plot(t, values_read[0,:])
    plt.show()

    # print(values_to_test)
    # print(values_read)
    # np.testing.assert_allclose(
    #     values_read, values_to_test, rtol=0.05, atol=0.005)
            
# with nidaqmx.Task() as write_task, nidaqmx.Task() as read_task:
#     write_task.ao_channels.add_ao_voltage_chan('Dev1/ao0',
#                                                max_val=10, min_val=-10)

#     read_task.ai_channels.add_ai_voltage_chan('Dev1/ai0',
#                                               max_val=10, min_val=-10)

#     writer = AnalogSingleChannelWriter(write_task.out_stream)
#     reader = AnalogSingleChannelReader(read_task.in_stream)

#     # Generate random values to test.
#     values_to_test = [random.uniform(-10, 10) for _ in range(len(array))]

#     values_read = []
#     for value_to_test in values_to_test:
#         writer.write_one_sample(value_to_test)
#         time.sleep(1./acq_freq)

#         value_read = reader.read_one_sample()
#         assert isinstance(value_read, float)
#         values_read.append(value_read)

#     np.testing.assert_allclose(
#         values_read, values_to_test, rtol=0.05, atol=0.005)
