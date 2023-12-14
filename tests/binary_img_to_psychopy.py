from psychopy import visual, core, event #import some libraries from PsychoPy
import time
import numpy as np
import matplotlib.pylab as plt

SCREEN = [1280, 720]
#create a window
mywin = visual.Window(SCREEN,
                      units="pix",
                      checkTiming=False,
                      fullscr=False)

protocol = 'quick-spatial-mapping'
props = np.load('../src/physion/acquisition/protocols/binaries/%s/protocol-0_index-0.npy' % protocol,
                allow_pickle=True).item()
array = np.fromfile('../src/physion/acquisition/protocols/binaries/%s/protocol-0_index-0.bin' % protocol,
                    dtype=np.uint8).reshape(props['binary_shape'])

plt.imshow(np.clip(2.*array[0,:,:]/255.-1, -1, 1))
plt.show()
print(mywin.size)
#draw the stimuli and update the window
i=0
while True: #this creates a never-ending loop
    stim = visual.ImageStim(win=mywin,
                            image=np.clip(2.*array[i,:,:].T/255.-1,
                                          -1, 1),
                            units='pix',
                            size=SCREEN)
    stim.draw()
    mywin.flip()
    i=np.min([i+1, array.shape[0]-1])

    if len(event.getKeys())>0:
        break
    event.clearEvents()

#cleanup
mywin.close()
core.quit()

