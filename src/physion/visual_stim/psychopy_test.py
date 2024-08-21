import os, sys, time
from psychopy import visual, core, event, monitors #import some libraries from PsychoPy
import numpy as np


if '.mp4' in sys.argv[-1]:

    # we play a mp4 movie

    SCREEN = [1280/2, 720/2]
    mywin = visual.Window(checkTiming=(os.name=='posix'),
                          units='pix',
                          fullscr=False)
    stim = visual.MovieStim(mywin, 
                            sys.argv[-1],
                            size=mywin.size/2,
                            units='pix')
    tic = time.time()
    stim.play()
    print(time.time()-tic)
    tic = time.time()

    while True: #this creates a never-ending loop
        stim.draw()
        mywin.flip()
        if len(event.getKeys())>0:
            break
        event.clearEvents()
        print(time.time()-tic)
        tic = time.time()

elif sys.argv[-1]=='grating':

    monitor = monitors.Monitor('Test', 
                               distance=15., width=60.)
    SCREEN = [int(1280/2), int(720/2)]
    monitor.setSizePix(SCREEN)
    #create a window
    mywin = visual.Window(SCREEN,
                          monitor=monitor,
                          checkTiming=(os.name=='posix'),
                          units='pix',
                          fullscr=False)

    stim = visual.GratingStim(win=mywin,
                              units='deg',
                              # mask='circle',
                              sf=0.04,
                              # ori=45,
                              size=(300,20))

    #draw the stimuli and update the window
    i=0
    while True: #this creates a never-ending loop

        stim.draw()
        mywin.flip()
        i+=1

        if len(event.getKeys())>0:
            break
        event.clearEvents()
else:

    SCREEN = [int(1280/2), int(720/2)]
    #create a window
    mywin = visual.Window(SCREEN,
                          units="pix",
                          checkTiming=(os.name=='posix'),
                          fullscr=False)

    X, Z = np.meshgrid(np.linspace(-1, 1, SCREEN[0]),
                       np.linspace(-1, 1, SCREEN[1]))

    #draw the stimuli and update the window
    i=0
    while True: #this creates a never-ending loop

        stim = visual.ImageStim(win=mywin,
                                image=np.clip(np.random.randn(*X.shape)+\
                                              np.sin(8*np.pi*X+i/10.),
                                              -1, 1),
                                # units='pix',
                                size=mywin.size)
        stim.draw()
        mywin.flip()
        i+=1

        if len(event.getKeys())>0:
            break
        event.clearEvents()

#cleanup
mywin.close()
core.quit()
