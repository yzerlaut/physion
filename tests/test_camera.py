"""
camera data (FaceCamera, RigCamera, ...) read from video files
"""
import logging
import numpy as np
import cv2 as cv

from physion.utils.camera import CameraData


def test_video_frame_warning_is_shown_once(tmp_path, caplog):
    fn = str(tmp_path/'FaceCamera.mp4')
    writer = cv.VideoWriter(fn, cv.VideoWriter_fourcc(*'mp4v'), 20, (32, 24))
    for i in range(10):
        writer.write(np.full((24, 32, 3), 10*i, dtype=np.uint8))
    writer.release()

    cam = CameraData.__new__(CameraData) # video-only object (no summary file)
    cam.name, cam.verbose = 'FaceCamera', False
    cam.binary_file, cam.FILES, cam.FRAMES = None, None, None
    cam.cap = cv.VideoCapture(fn)
    cam.nFrames = cam.nFrames_movie = int(cam.cap.get(cv.CAP_PROP_FRAME_COUNT))

    with caplog.at_level(logging.WARNING, logger='physion.utils.camera'):
        frames = [cam.get(i) for i in range(5)]

    assert all(f is not None for f in frames)
    assert len([r for r in caplog.records if 'precisely read' in r.getMessage()]) == 1
