import os, sys


#################################################
###        Select the Camera Interface    #######
#################################################
CameraInterface = None
### --------- MicroManager Interface -------- ###
try:
    from pycromanager import Core
    CameraInterface = 'MicroManager'
except ModuleNotFoundError:
    pass

### ------------ ThorCam Interface ---------- ###
if CameraInterface is None:
    try:
        absolute_path_to_dlls= os.path.join(os.path.expanduser('~'),
                            'work', 'physion', 'src', 'physion',
                            'hardware', 'Thorlabs', 'camera_dlls')
        os.environ['PATH'] = absolute_path_to_dlls + os.pathsep +\
                                                    os.environ['PATH']
        os.add_dll_directory(absolute_path_to_dlls)
        CameraInterface = 'ThorCam'
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    except (AttributeError, ModuleNotFoundError):
        pass

### --------- None -> demo mode ------------- ###
if CameraInterface is None:
    print('------------------------------------')
    print('   camera support not available !')
    print('------------------------------------')
    print('            ---> demo mode only !   ')
    print('------------------------------------')
