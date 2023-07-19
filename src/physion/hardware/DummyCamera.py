"""
Dummy Camera to be used with the 'multiprocessing' model
"""
import time, sys, os
import numpy as np
from pathlib import Path

class Acquisition:

    def __init__(self, 
                 settings={'frame_rate':30.}):
        self.running = False
        self.times = []
        self.img_size=(800, 600)
        self.settings = {}
        self.update_settings(settings)

    def update_settings(self,
                        settings={}):
        for key in settings:
            self.settings[key] = settings[key]
            print('updated: ', key, self.settings[key])

    def print_rec_infos(self):
        # some debugging infos here
        if len(self.times):
            print('%i frames recorded' % len(self.times))
            print('Camera -- effective sampling frequency: %.1f Hz ' %\
                            (1./np.mean(np.diff(self.times))))
        else:
            print('no frames recorded by the Camera')
        
    def stop(self):
        pass

    def rec_and_check(self, run_flag, quit_flag, folder,
                      debug=True):

        # # -- while loop 
        while not quit_flag.is_set():
           
            image= np.random.randn(*self.img_size).astype(np.uint16)
            Time = time.time()

            if not self.running and run_flag.is_set():
                # not running and need to start  !

                self.running, self.times = True, []

            elif self.running and not run_flag.is_set():
                # running and we need to stop

                self.running=False

                if debug:
                    self.print_rec_infos()

            # after the update
            if self.running:
                # we store the image and its timestamp

                np.save(os.path.join(folder.get(), '%s.npy' % Time), image)
                self.times.append(Time)
                time.sleep(1./self.settings['frame_rate'])

        # end of the while loop
        if debug:
            self.print_rec_infos()
        
        self.running=False
        self.stop()


def launch_Camera(run_flag, quit_flag, datafolder,
                  settings={'frame_rate':30.}):

    camera = Acquisition(settings=settings)
    camera.rec_and_check(run_flag, quit_flag, datafolder)


if __name__=='__main__':


    import multiprocessing
    from ctypes import c_char_p

    T=2

    run = multiprocessing.Event()
    quit_event = multiprocessing.Event()
    manager = multiprocessing.Manager()
    datafolder = manager.Value(c_char_p, 'datafolder')    

    camera_process = multiprocessing.Process(target=launch_Camera,
                                             args=(run, quit_event, datafolder))
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

