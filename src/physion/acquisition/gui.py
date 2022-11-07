from PyQt5 import QtWidgets, QtCore
import sys, time, os, pathlib, json
import numpy as np
import multiprocessing # for the camera streams !!

import physion

# if not sys.argv[-1]=='no-stim':
try:
    from physion.visual_stim.build import build_stim
    from physion.visual_stim.screens import SCREENS
except ModuleNotFoundError:
    print(' /!\ Problem with the Visual-Stimulation module /!\ ')
    SCREENS = []

try:
    from physion.hardware.NIdaq.main import Acquisition
except ModuleNotFoundError:
    print(' /!\ Problem with the NIdaq module /!\ ')

try:
    from physion.hardware.FLIRcamera.recording import launch_FaceCamera
except ModuleNotFoundError:
    print(' /!\ Problem with the FLIR camera module /!\ ')

def multimodal(self,
               tab_id=0):

    tab = self.tabs[tab_id]

    self.cleanup_tab(tab)





    self.refresh_tab(tab)

