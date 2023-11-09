import matplotlib.pylab as plt

import simple_pyspin, time, sys, os
import numpy as np
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

cam = simple_pyspin.Camera()
cam.init()
cam.start()
img = cam.get_array().astype(np.uint8)

plt.imshow(img)
plt.show()

