"""
https://pycro-manager.readthedocs.io/en/latest/core.html
This example shows how to use pycromanager to interact with the micro-manager core.
Aside from the setup section, each following section can be run independently
"""
from pycromanager import Core
import numpy as np

import time


#get object representing micro-manager core
core = Core()

#### Calling core functions ###
exposure = core.get_exposure()
print(exposure)
# print(dir(core))
# core.set_roi(0,100,0,100)
core.set_exposure(0.001)

#### Setting and getting properties ####
#Here we set a property of the core itself, but same code works for device properties
auto_shutter = core.get_property('Core', 'AutoShutter')
core.set_property('Core', 'AutoShutter', 0)

#### Acquiring images ####
#The micro-manager core exposes several mechanisms foor acquiring images. In order to
#not interfere with other pycromanager functionality, this is the one that should be used
N = 20
start = time.time()
for i in range(N):
    core.snap_image()
    tagged_image = core.get_tagged_image()
    #pixels by default come out as a 1D array. We can reshape them into an image
    pixels = np.reshape(tagged_image.pix,
                        newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
print('average frame rate over n=%i frames; %.1f (Hz)' % (N, N/(time.time()-start)))
