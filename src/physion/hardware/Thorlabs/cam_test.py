import os

absolute_path_to_dlls= os.path.join(os.path.expanduser('~'),
                                  'work', 'physion', 'src', 'physion',
                                  'hardware', 'Thorlabs', 'camera_dlls')
os.environ['PATH'] = absolute_path_to_dlls + os.pathsep + os.environ['PATH']
os.add_dll_directory(absolute_path_to_dlls)

from thorlabs_tsi_sdk.tl_camera import TLCameraSDK

import matplotlib.pylab as plt

with TLCameraSDK() as sdk:

    cameras = sdk.discover_available_cameras()
    if len(cameras) == 0:
        print("Error: no cameras detected!")
    elif len(cameras)==1:
        print(cameras[0])
        print("[ok] 1 camera found !")
    else:
        print(cameras)
        print("[ok] %i camera found ! \n   --> taking the first one !" % len(cameras))

    with sdk.open_camera(cameras[0]) as camera:
        #  setup the camera for continuous acquisition
        camera.frames_per_trigger_zero_for_unlimited = 0
        camera.image_poll_timeout_ms = 2000  # 2 second timeout
        camera.arm(2)

        # save these values to place in our custom TIFF tags later
        print('bit_depth ', camera.bit_depth)
        print('exposure', camera.exposure_time_us)

        # need to save the image width and height for color processing
        print('image_width', camera.image_width_pixels)
        print('image_height', camera.image_height_pixels)
