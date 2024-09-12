import os, sys, time
import cv2 as cv
import numpy as np

class Movie:

    def __init__(self,
                 movie_file,
                 loc=(255,150)):

        self.movie = cv.VideoCapture(movie_file)

        cv.namedWindow('frame', cv.WINDOW_NORMAL)
        cv.moveWindow('frame', *loc)
        # cv.resizeWindow('frame', 1280, 720)

        # cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        print(movie_file, 'initialized')

    def play(self):

        # Check if camera opened successfully
        if (self.movie.isOpened()== False): 
          print("Error opening video stream or file")
        
        t0, self.times = time.time(), []

        # Read until video is completed
        while self.movie.isOpened():

            tic = time.time()

            # Capture frame-by-frame
            #movie.set(2, 0.)
            ret, frame = self.movie.read()

            self.times.append(time.time()-tic)

            if ret == True:
         
                # Display the resulting frame
                cv.imshow('frame', frame)

                # Press Q on keyboard to  exit
                if cv.waitKey(25) & 0xFF == ord('q'):
                    break

            else:
                self.movie.set(cv.CAP_PROP_POS_FRAMES, 0)
                continue

        self.stop()

    def stop(self):
        # When everything done, release the video capture object
        self.movie.release()
        # Closes all the frames
        cv.destroyAllWindows()
        print(1e3*np.mean(self.times))

if len(sys.argv)>1:
    movie = sys.argv[-1]
else:
    movie = os.path.join(os.path.expanduser('~'), 'work', 'physion',
                        'src', 'physion', 'acquisition', 'protocols',
                        'movies', 'quick-spatial-mapping', 'movie.mp4')


m = Movie(movie)
m.play()
     

