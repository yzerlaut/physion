import numpy as np
import os, time
absolute_path_to_dlls= os.path.join(os.path.expanduser('~'),
                                  'work', 'physion', 'src', 'physion',
                                  'hardware', 'Thorlabs', 'camera_dlls')
os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
os.add_dll_directory(absolute_path_to_dlls)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK


def test(cam):

    cam.exposure_time_us = 100000
    cam.frames_per_trigger_zero_for_unlimited = 0
    cam.operation_mode = 0

    cam.arm(2) 
    cam.issue_software_trigger()
    t0 = time.time()
    frames, times = [], []
    while (time.time()-t0)<10:
        frame = cam.get_pending_frame_or_null()
        if frame is not None:
            frames.append(frame.image_buffer)
            times.append(time.time()-t0)
    print(frames[-1])
    print('effective frame rate: %.1f Hz' % (1./np.mean(np.diff(times))))
    cam.disarm()

if __name__ == "__main__":

    sdk = TLCameraSDK()
    camera_list = sdk.discover_available_cameras()
    cam = sdk.open_camera(camera_list[0])
    test(cam)
    test(cam)
    cam.dispose()
    sdk.dispose()



