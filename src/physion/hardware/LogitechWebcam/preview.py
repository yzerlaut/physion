import cv2

class stop_func: # dummy version of the multiprocessing.Event class
    def __init__(self):
        self.stop = False
    def set(self):
        self.stop = True
    def is_set(self):
        return self.stop

class RigView:

    def __init__(self, folder='./'):
        
        self.vc = cv2.VideoCapture(0)

        if self.vc.isOpened(): # try to get the first frame
            rval, self.frame = self.vc.read()
        else:
            rval = False

    def run(self, run_flag, stop_flag):
        while not stop_flag.is_set():
            cv2.imshow("Live view of experimental rig                                                  --   (ESC to close)", self.frame)
            rval, self.frame = self.vc.read()
            key = cv2.waitKey(20)
            if key == 27: # exit on ESC
                break
        cv2.destroyWindow("preview")


def launch_RigView(run_flag, quit_flag, folder):
    camera = RigView(folder=folder)
    camera.run(run_flag, quit_flag)
        
        
if __name__=='__main__':

    T = 5 # seconds

    import multiprocessing, time
    stop_event = multiprocessing.Event()
    RigView_process = multiprocessing.Process(target=launch_RigView, args=(stop_event,))
    RigView_process.start()
    print(stop_event.is_set())
    time.sleep(T/2)
    stop_event.set()
    print(stop_event.is_set())
    # camera.stop()


    
    

