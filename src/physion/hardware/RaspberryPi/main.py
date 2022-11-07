import sys, time

import RPi.GPIO as GPIO

sys.path.append(str(Path(__file__).resolve().parents[1]))
from zaber_stage import ZaberStageControl


class StageController:

    def __init__(self,
                 stage_controller_args={},
                 triggers = [],
                 modes = [],
                 actions = []):

        self.stage_controller = ZaberStageControl(**stage_controller_args)
        self.channels = channels
        self.modes = modes
        self.actions = actions
        
        GPIO.setwarnings(False) # Ignore warning for now
        GPIO.setmode(GPIO.BCM) # Use physical pin numbering
        for channel, mode, action in zip(self.channels, self.mode, self.actions):
            initiate_trigger(self, channel, mode, action)
        
        
    def initiate_trigger(self, channel, mode, action):
        if mode == 'in':
            GPIO.setup(channel, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
        elif mode == 'out':
            GPIO.setup(channel, GPIO.OUT)
        else:
            print('Unable to generate IO function')
        
        if mode == 'Rising':
            GPIO.add_event_detect(channel, GPIO.RISING)
        elif mode == 'Falling':
            GPIO.add_event_detect(channel, GPIO.FALLING)
        elif mode == 'Both':
            GPIO.add_event_detect(channel, GPIO.BOTH)
        else:
            print('Mode selected for channel {} incorrect.'.format(channel))
            print('Must be Rising, Falling or Both')
        
        if action == 'move stage':
            GPIO.add_event_callback(channel, self.stage_controller.stage_move)
        elif action == 'start_stop':
            GPIO.add_event_callback(channel, self.stage_controller.start_stop)  
    
    
    
    def stop_trigger(self, channel):
        GPIO.cleanup(channel)
    
    
    def stop_all(self):
        GPIO.cleanup()
    

if __name__=='__main__':

    print('- Setting up the controller')
    pi = StageController()
    time.sleep(2)
