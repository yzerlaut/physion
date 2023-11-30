import os
from psychopy import visual, core, event #import some libraries from PsychoPy
import numpy as np

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
