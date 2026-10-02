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
