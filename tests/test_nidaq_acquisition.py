"""
NIdaq acquisition: reading callbacks (NIdaq thread) vs "close" (GUI thread)

nidaqmx is replaced by a stub: the hardware is simulated by fake readers,
    the callbacks are called from another thread as by NI-DAQmx
"""
import sys, types, threading, time
import numpy as np
import pytest


@pytest.fixture
def nidaq(monkeypatch):
    """ physion.hardware.NIdaq modules with a stub of nidaqmx """
    stub = types.ModuleType('nidaqmx')
    stub.errors = types.SimpleNamespace(DaqError=type('DaqError', (Exception,), {}))
    stub.Task = object
    modules = {'nidaqmx': stub,
               'nidaqmx.utils': types.SimpleNamespace(flatten_channel_string=None),
               'nidaqmx.constants': types.SimpleNamespace(Edge=None, WAIT_INFINITELY=-1, ProductCategory=None,
                                                          LineGrouping=None),
               'nidaqmx.stream_readers': types.SimpleNamespace(
                   AnalogMultiChannelReader=None, DigitalMultiChannelReader=None),
               'nidaqmx.stream_writers': types.SimpleNamespace(
                   AnalogMultiChannelWriter=None, DigitalMultiChannelWriter=None)}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    for module in ['main', 'usb']:
        monkeypatch.delitem(sys.modules, 'physion.hardware.NIdaq.%s' % module, raising=False)
    from physion.hardware.NIdaq import main, usb
    return types.SimpleNamespace(main=main, usb=usb)


@pytest.fixture
def Acquisition(nidaq):
    return nidaq.main.Acquisition


class FakeTask:
    def close(self): pass


class FakeReaders:
    """ analog channel 0 and digital line 2 carry the sample index (to check the order) """
    def __init__(self, delay=0., blocking=None):
        self.n, self.delay, self.blocking = 0, delay, blocking

    def read_many_sample(self, buffer, num_samples, timeout=None):
        if self.blocking is not None:
            self.blocking.wait()
        time.sleep(self.delay)
        buffer[0,:] = self.n+np.arange(num_samples)

    def read_many_sample_port_uint32(self, buffer, num_samples, timeout=None):
        buffer[0,:] = ((self.n+np.arange(num_samples))%2) << 2
        self.n += num_samples


def running_acquisition(Acquisition, tmp_path, readers):
    """ an Acquisition as after "launch", without hardware """
    acq = Acquisition.__new__(Acquisition)
    acq.lock = threading.Lock()
    acq.analog_buffers, acq.digital_buffers = [], []
    acq.dt, acq.Nchannel_analog_in = 1e-3, 2
    acq.filename = str(tmp_path/'NIdaq.npy')
    acq.analog_data = np.zeros((2, 0))
    acq.digital_data = np.zeros((1, 0), dtype=np.uint8)
    acq.digital_in_chan = 'Dev1/port0/line2:3'
    for key in ['read_analog_task', 'read_digital_task', 'write_analog_task',
                'write_digital_task', 'sample_clk_task']:
        setattr(acq, key, FakeTask())
    acq.analog_reader = acq.digital_reader = readers
    acq.running, acq.data_saved = True, False
    return acq


def check_saved_data(acq, n_buffers, buffer_size):
    data = np.load(acq.filename, allow_pickle=True).item()
    n = n_buffers*buffer_size
    assert data['analog'].shape == (2, n) and data['digital'].shape == (2, n)
    np.testing.assert_array_equal(data['analog'][0], np.arange(n))
    np.testing.assert_array_equal(data['digital'][0], np.arange(n)%2)   # line 2
    assert not data['digital'][1].any()                                  # line 3


def test_close_waits_for_the_callback_in_progress(Acquisition, tmp_path):
    """ the race of the end of acquisition: a callback reading while "close" is called """
    readers = FakeReaders()
    acq = running_acquisition(Acquisition, tmp_path, readers)
    acq.reading_task_callback(0, 0, 100)
    readers.blocking = blocking = threading.Event()   # the next read waits
    callback = threading.Thread(target=acq.reading_task_callback, args=(0, 0, 100))
    callback.start()                     # blocked in the analog read
    closing = threading.Thread(target=acq.close)
    closing.start()
    time.sleep(0.2)
    assert closing.is_alive()            # "close" waits for the callback ...
    blocking.set()                       # ... until its buffer is read
    callback.join(); closing.join()
    check_saved_data(acq, n_buffers=2, buffer_size=100)
    # callbacks after "close" do nothing
    acq.reading_task_callback(0, 0, 100)
    assert len(acq.analog_buffers) == 0


def test_close_during_continuous_callbacks(Acquisition, tmp_path):
    acq = running_acquisition(Acquisition, tmp_path, FakeReaders(delay=1e-3))
    def nidaq_thread():
        for i in range(1000):
            acq.reading_task_callback(0, 0, 50)
    thread = threading.Thread(target=nidaq_thread)
    thread.start()
    time.sleep(0.1)
    acq.close()
    thread.join()
    n_buffers = len(np.load(acq.filename, allow_pickle=True).item()['analog'][0])//50
    assert n_buffers > 0
    check_saved_data(acq, n_buffers, buffer_size=50)


def test_analog_and_digital_cut_to_the_same_length(Acquisition, tmp_path):
    acq = running_acquisition(Acquisition, tmp_path, FakeReaders())
    acq.reading_task_callback(0, 0, 100)
    acq.analog_buffers.append(np.zeros((2, 100)))    # one more analog buffer
    acq.close()
    check_saved_data(acq, n_buffers=1, buffer_size=100)


########################################################
#   USB cards (physion.hardware.NIdaq.usb)
########################################################

def running_usb_acquisition(nidaq, tmp_path, readers):
    """ a USB Acquisition as after "launch", without hardware """
    acq = nidaq.usb.Acquisition.__new__(nidaq.usb.Acquisition)
    acq.lock = threading.Lock()
    acq.analog_buffers, acq.digital_buffers = [], []
    acq.dt, acq.Nchannel_analog_in, acq.Nchannel_digital_in = 1e-3, 2, 1
    acq.filename, acq.outputs = str(tmp_path/'NIdaq.npy'), None
    acq.analog_data = np.zeros((2, 0))
    acq.digital_data = np.zeros((1, 0), dtype=np.uint32)
    acq.read_analog_task = acq.read_digital_task = acq.sample_clk_task = FakeTask()
    acq.analog_reader = acq.digital_reader = readers
    acq.running, acq.data_saved = True, False
    return acq


def test_usb_close_waits_for_the_callback_in_progress(nidaq, tmp_path):
    readers = FakeReaders()
    acq = running_usb_acquisition(nidaq, tmp_path, readers)
    acq.reading_task_callback(0, 0, 100)
    readers.blocking = blocking = threading.Event()   # the next read waits
    callback = threading.Thread(target=acq.reading_task_callback, args=(0, 0, 100))
    callback.start()
    closing = threading.Thread(target=acq.close)
    closing.start()
    time.sleep(0.2)
    assert closing.is_alive()
    blocking.set()
    callback.join(); closing.join()
    analog, digital, dt = acq.close(return_data=True)
    assert analog.shape == (2, 200) and digital.shape == (1, 200)
    np.testing.assert_array_equal(analog[0], np.arange(200))
    np.testing.assert_array_equal(digital[0], (np.arange(200)%2) << 2)
    saved = np.load(acq.filename, allow_pickle=True).item()
    np.testing.assert_array_equal(saved['analog'], analog)
    acq.reading_task_callback(0, 0, 100)
    assert len(acq.analog_buffers) == 0


class FakeZeroHardware:
    """ on-demand output tasks, what they write and when """

    def __init__(self, main, monkeypatch, failing=False):
        self.events, self.failing = [], failing
        hardware = self

        class Channels:
            def __init__(self, kind):
                self.kind = kind
            def add_do_chan(self, chan, line_grouping=None):
                hardware.events.append(('digital-chan', chan))
            def add_ao_voltage_chan(self, chan, max_val=None, min_val=None):
                hardware.events.append(('analog-chan', chan))

        class Task:
            def __init__(self):
                self.do_channels, self.ao_channels = Channels('do'), Channels('ao')
                self.out_stream = None
            def __enter__(self):
                if hardware.failing:
                    raise RuntimeError('device busy')
                return self
            def __exit__(self, *args):
                hardware.events.append('task-closed')

        class DigitalWriter:
            def __init__(self, stream): pass
            def write_one_sample_port_uint32(self, data):
                assert data.dtype==np.uint32 and data.ndim==1
                hardware.events.append(('digital-write', data.tolist()))

        class AnalogWriter:
            def __init__(self, stream): pass
            def write_one_sample(self, data):
                assert data.dtype==np.float64 and data.ndim==1
                hardware.events.append(('analog-write', data.tolist()))

        monkeypatch.setattr(main.nidaqmx, 'Task', Task)
        monkeypatch.setattr(main, 'DigitalMultiChannelWriter', DigitalWriter)
        monkeypatch.setattr(main, 'AnalogMultiChannelWriter', AnalogWriter)
        monkeypatch.setattr(main, 'flatten_channel_string', ', '.join)
        monkeypatch.setattr(main, 'LineGrouping', types.SimpleNamespace(CHAN_FOR_ALL_LINES='all'))


class RecordedTask:
    def __init__(self, name, events):
        self.name, self.events = name, events
    def close(self):
        self.events.append('closed-'+self.name)


def acquisition_with_outputs(Acquisition, tmp_path, hardware):
    acq = running_acquisition(Acquisition, tmp_path, FakeReaders())
    acq.digital_out_chan, acq.output_channels = 'Dev1/port0/line3:7', ['Dev1/ao0', 'Dev1/ao1']
    for key in ['write_digital_task', 'write_analog_task', 'sample_clk_task']:
        setattr(acq, key, RecordedTask(key, hardware.events))
    return acq


def test_close_sets_all_outputs_to_zero(nidaq, tmp_path, monkeypatch):
    hardware = FakeZeroHardware(nidaq.main, monkeypatch)
    acq = acquisition_with_outputs(nidaq.main.Acquisition, tmp_path, hardware)
    acq.reading_task_callback(0, 0, 100)
    acq.close()
    assert ('digital-chan', 'Dev1/port0/line3:7') in hardware.events
    assert ('digital-write', [0]) in hardware.events
    assert ('analog-chan', 'Dev1/ao0, Dev1/ao1') in hardware.events
    assert ('analog-write', [0., 0.]) in hardware.events
    # the lines can only be written once the acquisition tasks released them
    first_write = min(hardware.events.index(('digital-chan', 'Dev1/port0/line3:7')),
                      hardware.events.index(('analog-chan', 'Dev1/ao0, Dev1/ao1')))
    assert hardware.events.index('closed-sample_clk_task') < first_write
    check_saved_data(acq, 1, 100)
    # only once
    n = len(hardware.events)
    acq.close()
    assert len(hardware.events) == n


def test_data_are_saved_if_the_outputs_can_not_be_set_to_zero(nidaq, tmp_path, monkeypatch, capsys):
    hardware = FakeZeroHardware(nidaq.main, monkeypatch, failing=True)
    acq = acquisition_with_outputs(nidaq.main.Acquisition, tmp_path, hardware)
    acq.reading_task_callback(0, 0, 100)
    acq.close()
    out = capsys.readouterr().out
    assert 'digital outputs NOT set to zero' in out and 'analog outputs NOT set to zero' in out
    check_saved_data(acq, 1, 100)


def test_no_output_task_nothing_to_set_to_zero(nidaq, tmp_path, monkeypatch):
    hardware = FakeZeroHardware(nidaq.main, monkeypatch)
    acq = acquisition_with_outputs(nidaq.main.Acquisition, tmp_path, hardware)
    acq.write_digital_task = acq.write_analog_task = None
    acq.close()
    assert not any(isinstance(e, tuple) for e in hardware.events)
