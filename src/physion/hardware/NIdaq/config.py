import nidaqmx

from nidaqmx.constants import ProductCategory, UsageTypeAI

acq_freq = 1000. # seconds

def get_analog_input_channels(device):
    return  [c.name for c in device.ai_physical_chans]

def get_digital_input_channels(device):
    return  [c.name for c in device.di_lines]

def get_counter_input_channels(device):
    return  [c.name for c in device.co_physical_chans]

def get_analog_output_channels(device):
    return  [c.name for c in device.ao_physical_chans]

def find_x_series_devices():
    system = nidaqmx.system.System.local()

    DEVICES = []
    for device in system.devices:
        # if (not device.dev_is_simulated and
        if (not device.is_simulated and
                device.product_category == ProductCategory.X_SERIES_DAQ and
                len(device.ao_physical_chans) >= 2 and
                len(device.ai_physical_chans) >= 4 and
                len(device.do_lines) >= 8 and
                (len(device.di_lines) == len(device.do_lines)) and
                len(device.ci_physical_chans) >= 4):
            DEVICES.append(device)
    return DEVICES

def find_m_series_devices():
    system = nidaqmx.system.System.local()

    DEVICES = []
    for device in system.devices:
        # if (not device.dev_is_simulated and
        if (not device.is_simulated and
                device.product_category == ProductCategory.M_SERIES_DAQ and
                len(device.ao_physical_chans) >= 2 and
                len(device.ai_physical_chans) >= 4):
            DEVICES.append(device)
    return DEVICES

def find_usb_devices():
    system = nidaqmx.system.System.local()

    DEVICES = []
    for device in system.devices:
        # if (not device.dev_is_simulated and
        if (not device.is_simulated and
                device.product_category == ProductCategory.USBDAQ):
            DEVICES.append(device)
    return DEVICES

if __name__=='__main__':

    print('----------------------')
    print('looking for M-series devices [...]')
    DEVICES = find_m_series_devices()
    print(DEVICES)
    print()
    print('----------------------')
    print('looking for X-series devices [...]')
    DEVICES = find_x_series_devices()
    print(DEVICES)
    print()
    print('----------------------')
    print('looking for USB devices [...]')
    DEVICES = find_usb_devices()
    print(DEVICES)
    print()


    # device = DEVICES[0]
    # print(dir(device))
    system = nidaqmx.system.System.local()

    DEVICES = []
    for device in system.devices:
        print('------------------------------------')
        print(device, device.product_category)
        print('Analog Input channels:')
        print(get_analog_input_channels(device))
        print('Digital Input channels:')
        print(get_digital_input_channels(device))
        print('Counter Input channels:')
        print(get_counter_input_channels(device))
        print('Analog Output channels:')
        print(get_analog_output_channels(device))
