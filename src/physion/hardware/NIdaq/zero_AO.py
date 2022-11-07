"""
http://zone.ni.com/reference/en-XX/help/370469AP-01/daqmxprop/attr2255/
"""

import nidaqmx, time
import numpy as np

from nidaqmx.utils import flatten_channel_string
from nidaqmx.constants import Edge
from nidaqmx.stream_readers import (
    AnalogSingleChannelReader, AnalogMultiChannelReader)
from nidaqmx.stream_writers import (
    AnalogSingleChannelWriter, AnalogMultiChannelWriter)

import sys, os, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))
from hardware_control.NIdaq.config import find_x_series_devices, find_m_series_devices, get_analog_input_channels, get_analog_output_channels

if len(sys.argv)==3:
    channel, value = int(sys.argv[1]), float(sys.argv[2])
    system = nidaqmx.system.System.local()
    device = system.devices[0]

    AOchannels = get_analog_output_channels(device)

    chan = get_analog_output_channels(device)[0]

    with nidaqmx.Task() as task:
        task.ao_channels.add_ao_voltage_chan(chan)
        task.ao_channels.ao_dac_offset_val(value)
    
else:
    print("""

    Should be used as:
                       python zero_AO.py channel value,

    e.g. 'python zero_AO.py 0 3.4'
    """)

