# National Instruments DAQ cards

*control of National Instruments DAQ cards for data acquisition and stimulation*

## Install

- Install NIdaq-mx, get it from the National Instruments website:
  - either https://www.ni.com/fr-fr/support/downloads/drivers/download.ni-daqmx.html#348669
  - or https://www.ni.com/fr-fr/support/downloads/drivers/download.ni-daqmx.html#291872

- Install the python API:
  `pip install nidaqmx`


## Code reference

- https://nidaqmx-python.readthedocs.io/en/latest/

But the `python` API isn't so well documented... This is a translation of the `C` API that is well documented.

There are a few example scripts:
- https://github.com/ni/nidaqmx-python/tree/master/nidaqmx_examples

But the present code is actually based on the material available in the tests:

- https://github.com/ni/nidaqmx-python/tree/master/nidaqmx/tests

In particular the file:

- https://github.com/ni/nidaqmx-python/blob/master/nidaqmx/tests/test_stream_analog_readers_writers.py