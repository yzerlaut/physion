import os, time

from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units


class ZaberStageControl:
    """
    
    """
    def __init__(self,
                 comp_port=None,
                 position_start = 0,
                 offset=10):

        if comp_port is None:
            # COM port
            if os.name == 'nt':
                com_port = 'COM4'
            elif os.uname()[4][:3] == 'arm':
                com_port = '/dev/ttyUSB0'
            else:
                raise Exception('', 'Need to specify port')

        self.offset = offset
        self.position_whiskers = 0
        self.current_position = 0

        self.status = 'Standby'
        
        Library.enable_device_db_store()
        self.connection = Connection.open_serial_port(com_port)
        self.init_device()

        
    def init_device(self):
        # Setup and home x-axis
        device_list = self.connection.detect_devices()
        print("    -Found {} devices".format(len(device_list)))
        self.device = device_list[0]
        self.axis = self.device.get_axis(1)
        print("    -Homing the device")
        time.sleep(1)
        self.axis.home()
        time.sleep(1)
        
        
    def set_start_position(self):
        self.axis.home()
        time.sleep(2)
        print('Using the manual controls, set the stimulator in the whisker field')
        time.sleep(1)
        input("Press Enter when ready...")
        self.position_whiskers = self.axis.get_position(Units.LENGTH_MILLIMETRES)
        self.axis.move_absolute((self.position_whiskers - self.offset), Units.LENGTH_MILLIMETRES)


    def start_stop(self):
        if self.status == 'Standby':
            print('\n- Experiment Started')
            self.status = 'Recording'
    
        elif self.status == 'Recording':
            print('\n- Experiment Stopped')
            self.status = 'Standby'
            self.axis.move_absolute((self.position_whiskers - self.offset), Units.LENGTH_MILLIMETRES)
            
        
    def move(self):
        if self.status == 'Recording':
            self.current_position = self.axis.get_position()
            if self.current_position == self.position_whiskers:
                self.axis.move_absolute((self.position_whiskers - self.offset), Units.LENGTH_MILLIMETRES)
            elif self.current_position == (self.position_whiskers - self.offset):
                self.axis.move_absolute(self.position_whiskers, Units.LENGTH_MILLIMETRES)
            else:
                print('* Stage position is not correct')
        else:
            print('* Recording not started')
            
    def close(self):
        print('* Closing connection to Zaber Stage....')
        self.connection.close()


if __name__=='__main__':

    control = ZaberStageControl()