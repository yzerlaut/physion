import clr
import time
from thorcam.camera import ThorCam

class MyThorCam(ThorCam):
    def received_camera_response(self, msg, value):
        super(MyThorCam, self).received_camera_response(msg, value)
        if msg == 'image':
            return
        print('Received "{}" with value "{}"'.format(msg, value))
    def got_image(self, image, count, queued_count, t):
        print('Received image "{}" with time "{}" and counts "{}", "{}"'
              .format(image, t, count, queued_count))

cam = MyThorCam()

cam.start_cam_process()

cam.refresh_cameras()

print('initializing camera [...]')
time.sleep(5)

cam.open_camera('21373')

print('exposure range', cam.exposure_range)

cam.play_camera()

cam.stop_playing_camera()

# close the camera
cam.close_camera()

# close the server and everything
cam.stop_cam_process(join=True)

