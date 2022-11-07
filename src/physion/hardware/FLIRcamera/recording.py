"""

"""
import simple_pyspin, time, sys, os
from skimage.io import imsave
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
desktop_png = os.path.join(os.path.expanduser("~/Desktop"), 'FaceCamera.png')

class stop_func: # dummy version of the multiprocessing.Event class
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop
    
class CameraAcquisition:

    def __init__(self,
                 settings={'frame_rate':20.}):
        
        self.times, self.running = [], False
        self.init_camera(settings)

    def init_camera(self, settings):
        
        self.cam = simple_pyspin.Camera()
        self.cam.init()

        ########################################################################
        # -------------------------------------------------------------------- #
        ## -- SETTINGS through the FlyCap or SpinView software, easier....  ####
        # -------------------------------------------------------------------- #
        ########################################################################

        
        # # Set the area of interest (AOI) to the middle half
        # self.cam.Width = self.cam.SensorWidth // 2
        # self.cam.Height = self.cam.SensorHeight // 2
        # self.cam.OffsetX = self.cam.SensorWidth // 4
        # self.cam.OffsetY = self.cam.SensorHeight // 4

        # # To change the frame rate, we need to enable manual control
        # self.cam.AcquisitionFrameRateAuto = 'Off'
        # # self.cam.AcquisitionFrameRateEnabled = True # seemingly not available here
        # self.cam.AcquisitionFrameRate = settings['frame_rate']

        # # To control the exposure settings, we need to turn off auto
        # self.cam.GainAuto = 'Off'
        # # Set the gain to 20 dB or the maximum of the camera.
        # max_gain = self.cam.get_info('Gain')['max']
        # if (settings['gain']==0) or (settings['gain']>max_gain):
        #     self.cam.Gain = max_gain
        #     print("Setting FaceCamera gain to %.1f dB" % max_gain)
        # else:
        #     self.cam.Gain = settings['gain']

        # self.cam.ExposureAuto = 'Off'
        # self.cam.ExposureTime =settings['exposure_time'] # microseconds, ~20% of interframe interval


    def save_sample_on_desktop(self):
        ### SAVING A SAMPLE ON THE DESKTOP
        print('saving a sample image as:', desktop_png)
        imsave(desktop_png, np.array(self.cam.get_array()))

    def rec_and_check(self, run_flag, quit_flag, folder):
        
        self.cam.start()
        self.save_sample_on_desktop()
        
        while not quit_flag.is_set():
            
            if not self.running and run_flag.is_set() : # not running and need to start  !
                self.save_sample_on_desktop()
                self.running, self.times = True, []
                # reinitialize recording
                self.imgs_folder = os.path.join(folder.get(), 'FaceCamera-imgs')
                Path(self.imgs_folder).mkdir(parents=True, exist_ok=True)
            elif self.running and not run_flag.is_set(): # running and we need to stop
                self.running=False
                print('FaceCamera -- effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(self.times))))
                self.save_sample_on_desktop()
                

            # after the update
            if self.running:
                image, Time = self.cam.get_array().astype(np.uint8), time.time()
                np.save(os.path.join(self.imgs_folder, '%s.npy' % Time), image)
                # imsave(os.path.join(self.imgs_folder, '%s.png' % Time), np.array(image).astype(np.uint8)) # TOO SLOW
                self.times.append(Time)

        if len(self.times)>0:
            print('FaceCamera -- effective sampling frequency: %.1f Hz ' % (1./np.mean(np.diff(self.times))))
            self.save_sample_on_desktop()
        
        self.running=False
        self.cam.stop()

def launch_FaceCamera(run_flag, quit_flag, datafolder,
                      settings={'frame_rate':20.}):
    camera = CameraAcquisition(settings=settings)
    camera.rec_and_check(run_flag, quit_flag, datafolder)

    
if __name__=='__main__':

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    T = 2 # seconds

    import multiprocessing
    from ctypes import c_char_p
    
    run = multiprocessing.Event()
    quit_event = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')    
    camera_process = multiprocessing.Process(target=launch_FaceCamera, args=(run, quit_event, datafolder))
    run.clear()
    camera_process.start()

    # start first acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '1')))
    run.set()
    time.sleep(T)
    run.clear()

    # start second acq
    datafolder.set(str(os.path.join(os.path.expanduser('~'), 'DATA', '2')))
    run.set()
    time.sleep(T)
    run.clear()
    time.sleep(0.5)
    # quit process
    quit_event.set()
