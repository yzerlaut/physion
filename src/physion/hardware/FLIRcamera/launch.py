import sys, os
from recording import CameraAcquisition

if len(sys.argv)==1:
    print("""
    should be used as :
       python launch.py 30 test1 # for putting the images in the folder named "test1"
    """)
else:
    duration = float(sys.argv[1])
    folder = sys.argv[2]
    if os.path.isdir(folder):
        print(folder,  ' already exists pick another one')
    else:
        os.mkdir(folder)
        camera = CameraAcquisition(folder=folder)
        camera.rec(duration)

