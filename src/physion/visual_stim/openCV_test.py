import os, sys, time
import cv2
import numpy as np


if '.mp4' in sys.argv[-1]:

    movie = cv2.VideoCapture(sys.argv[-1])

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.moveWindow('frame', 0, -700)
    cv2.resizeWindow('frame', 1280, 720)

    cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Check if camera opened successfully
    if (movie.isOpened()== False): 
      print("Error opening video stream or file")
    
    times = []
    # Read until video is completed
    while(movie.isOpened()):

      tic = time.time()
      # Capture frame-by-frame
      #movie.set(2, 0.)
      ret, frame = movie.read()

      times.append(time.time()-tic)

      if ret == True:
     
        # Display the resulting frame
        cv2.imshow('frame', frame)
     
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
      # Break the loop
      else: 
        break

     
    # When everything done, release the video capture object
    movie.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()

    print(1e3*np.mean(times))
else:
    print('need to provide a mp4 as argument !')

