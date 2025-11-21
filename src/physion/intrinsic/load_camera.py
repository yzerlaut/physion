import os, sys


#################################################
###        Select the Camera Interface    #######
#################################################
CameraInterface = None
### --------- MicroManager Interface -------- ###
try:
    # pip uninstall pycromanager to be sure not use this camera
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
    except BaseException as be:
        # print(be)
        pass

### --------- None -> demo mode ------------- ###
if CameraInterface is None:
    print()
    print()
    print('------------------------------------')
    print('   camera support not available !')
    print('------------------------------------')
    print('            ---> demo mode only !   ')
    print('------------------------------------')
    print()

camera_depth = 12 # 12-bit camera depth


############################################
#   --       ThorCam Functions .      --   #
############################################

def init_thorlab_cam(self):
    self.sdk = TLCameraSDK()
    self.cam = self.sdk.open_camera(self.sdk.discover_available_cameras()[0])
    # for software trigger
    self.cam.frames_per_trigger_zero_for_unlimited = 0
    self.cam.operation_mode = 0
    print('\n [ok] Thorlabs Camera successfully initialized ! \n')
    self.demo = False

def close_thorlab_cam(self):
    self.cam.dispose()
    self.sdk.dispose()
