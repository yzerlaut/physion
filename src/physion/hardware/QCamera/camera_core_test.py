"""
https://pycro-manager.readthedocs.io/en/latest/core.html
This example shows how to use pycromanager to interact with the micro-manager core.
Aside from the setup section, each following section can be run independently
"""
from pycromanager import Core
import numpy as np
import matplotlib.pyplot as plt

#Setup
# get object representing MMCore
core = Core()

#### Calling core functions ###
exposure = core.get_exposure()
print(exposure)

#### Acquiring images ####
#The micro-manager core exposes several mechanisms foor acquiring images. In order to
#not interfere with other pycromanager functionality, this is the one that should be used
core.snap_image()
tagged_image = core.get_tagged_image()
#If using micro-manager multi-camera adapter, use core.getTaggedImage(i), where i is
#the camera index

#pixels by default come out as a 1D array. We can reshape them into an image
pixels = np.reshape(tagged_image.pix,
                        newshape=[tagged_image.tags['Height'], tagged_image.tags['Width']])
#plot it
plt.imshow(pixels, cmap='gray')
plt.show()

