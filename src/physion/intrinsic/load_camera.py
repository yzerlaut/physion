import os, sys, time

#################################################
###        Select the Camera Interface    #######
#################################################
if len(sys.argv)>2:
    # i.e. we passed the camera as argument: "python -m physion intrinsic Toupcam"
    CameraInterface = sys.argv[2]
else:
    CameraInterface = None
    camera_depth = 8


### --------- MicroManager Interface -------- ###
if CameraInterface == 'MicroManager':
    camera_depth = 12 

### ------------ Toupcam Interface ---------- ###
# if CameraInterface == 'Toupcam':
#     try:
#         from physion.hardware.toupcam import toupcam
#         print('starting to load Toupcam interface [...]')
#         toupcam.Toupcam.EnumV2()  # force DLL load
#         CameraInterface = 'Toupcam'
#         camera_depth = 8
#     except BaseException as be:
#         CameraInterface = None


### ------------ ThorCam Interface ---------- ###
if CameraInterface == 'ThorCam':
    try:
        absolute_path_to_dlls= os.path.join(os.path.expanduser('~'),
                            'work', 'physion', 'src', 'physion',
                            'hardware', 'Thorlabs', 'camera_dlls')
        os.environ['PATH'] = absolute_path_to_dlls + os.pathsep +\
                                                    os.environ['PATH']
        os.add_dll_directory(absolute_path_to_dlls)
        CameraInterface = 'ThorCam'
        camera_depth = 12
        from thorlabs_tsi_sdk.tl_camera import TLCameraSDK
    except BaseException as be:
        CameraInterface = None

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

############################################
#   --       Toupcam Functions .      --   #
############################################

def init_toupcam(self, warmup_seconds=3.0):
    # Enumerate cameras
    cams = toupcam.Toupcam.EnumV2()
    if not cams:
        raise RuntimeError("No Toupcam camera found")

    self.cam_info = cams[0]
    self.cam = toupcam.Toupcam.Open(self.cam_info.id)
    if not self.cam:
        raise RuntimeError("Failed to open Toupcam")

    # Resolution
    self.res = self.cam.get_eSize()
    self.width = self.cam_info.model.res[self.res].width
    self.height = self.cam_info.model.res[self.res].height

    # RGB byte order (important)
    self.cam.put_Option(toupcam.TOUPCAM_OPTION_BYTEORDER, 0)

    # Auto exposure (ThorCam equivalent: continuous mode)
    self.cam.put_AutoExpoEnable(1)

    # Allocate image buffer (RGB888)
    self.stride = toupcam.TDIBWIDTHBYTES(self.width * 24)
    self.buf = bytes(self.stride * self.height)

    # Start camera in pull mode (no callback)
    self.cam.StartPullModeWithCallback(None, None)

    # Warm-up (critical!)
    t0 = time.time()
    while time.time() - t0 < warmup_seconds:
        try:
            self.cam.PullImageV4(self.buf, 0, 24, 0, None)
        except toupcam.HRESULTException:
            pass
        time.sleep(0.02)

    print('\n [ok] Toupcam successfully initialized ! \n')
    self.demo = False

def close_toupcam(self):
    if hasattr(self, "cam") and self.cam:
        self.cam.Close()
        self.cam = None

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
