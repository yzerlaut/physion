import os, sys, time
import cv2 as cv
import numpy as np

def launch_VisualStim(movie_file):

    movie = cv.VideoCapture(movie_file)

    cv.namedWindow('frame', cv.WINDOW_NORMAL)
    # cv.moveWindow('frame', 0, -700)
    # cv.resizeWindow('frame', 1280, 720)

    # cv.setWindowProperty('frame', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

    # Check if camera opened successfully
    if (movie.isOpened()== False): 
      print("Error opening video stream or file")
    
    t0, times = time.time(), []

    # Read until video is completed
    while movie.isOpened():

        tic = time.time()

        # Capture frame-by-frame
        #movie.set(2, 0.)
        ret, frame = movie.read()

        times.append(time.time()-tic)

        if ret == True:
     
            # Display the resulting frame
            cv.imshow('frame', frame)

            # Press Q on keyboard to  exit
            if cv.waitKey(25) & 0xFF == ord('q'):
                break

    # When everything done, release the video capture object
    movie.release()
     
    # Closes all the frames
    cv.destroyAllWindows()

    print(1e3*np.mean(times))
     

if '.mp4' in sys.argv[-1]:

    launch_VisualStim(sys.argv[-1])
     
else:
    print('need to provide a mp4 as argument !')

