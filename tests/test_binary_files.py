from psychopy import visual, core, event #import some libraries from PsychoPy
import numpy as np

SCREEN = [1280, 720]
#create a window
mywin = visual.Window(SCREEN,
                      monitor="testMonitor", 
                      units="pix",
                      checkTiming=False,
                      fullscr=False)

props = np.load('../src/physion/acquisition/protocols/binaries/drifting-gratings/protocol-0_index-0.npy',
                allow_pickle=True).item()
array = np.fromfile('../src/physion/acquisition/protocols/binaries/drifting-gratings/protocol-0_index-0.bin',
                    dtype=np.uint8).reshape(props['binary_shape'])

#draw the stimuli and update the window
i=0
while True: #this creates a never-ending loop
    stim = visual.ImageStim(win=mywin,
                            image=np.clip(2.*array[i,:,:]/255.-1,
                                          -1, 1),
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
