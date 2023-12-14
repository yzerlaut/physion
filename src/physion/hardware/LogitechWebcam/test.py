from threading import Thread
import cv2, time

#again define resolution and fps
x=int(1920/2)
y=int(1080/2)
fps=30

class VideoStreamWidget(object):
    def __init__(self, src=2):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(3,int(x))
        self.capture.set(4,int(y))
        self.capture.set(cv2.CAP_PROP_FPS, int(fps))
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            time.sleep(.03)

    def show_frame(self):
        out.write(self.frame)
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            out.release()
            cv2.destroyAllWindows()
            exit(1)

if __name__ == '__main__':
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter("output.avi",fourcc, fps, (x,y))
    video_stream_widget = VideoStreamWidget()
    while True:
        try:
            video_stream_widget.show_frame()
        except AttributeError:
            pass
