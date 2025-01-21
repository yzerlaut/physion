"""
two studies for params:

 --- From Stringer et al., Nature 2019
Sparse noise stimuli consisted of white or black squares on a gray background. Squares were
of size 5°, and changed intensity every 200 ms. On each frame, the intensity of each square
was chosen independently, as white with 2.5% probability, black with 2.5% probability, and
gray with 95% probability. The sparse noise movie contained 6000 frames, lasting 20
minutes, and the same movie was played twice to allow cross-validated analysis.

 --- From Smith & Hausser, Nat. Neurosci. 2010 
Sparse-noise visual stimuli consisted of black (2 cd m−2) and white
(86 cd m−2) dots on a gray (40 cd m−2) background15. Dots ranged in size from
1.3–8.0° in diameter. Adjacent frames had no overlap in dot locations so that all
pixel transitions were to or from gray, never from black to white or vice versa.
Single dots. At 3Hz change frequency. 

"""
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    SPARSE NOISE   --- #####
####################################

params = {"movie_refresh_freq":30.,
          "presentation-duration":0.5,
          "presentation-interstim-period":0,
          "screen-width (deg)":116,
          "screen-height (deg)":84,
          "size (deg)":6.,
          "sparseness (%)":4,
          "black-white-ratio (%)":50,
          "seed (#)":1,
          # "shape ":0, # 0 is square, 1 is circle
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 'size'])

        # screen size 
        Dx = protocol["screen-width (deg)"]
        Dy = protocol["screen-height (deg)"]

        # calculate number of dots:
        self.ndots = int(\
                np.floor(((Dx*Dy) * protocol['sparseness (%)']/100. /\
                        np.max(self.experiment['size'])**2 ) / 2.) * 2.) 
        iThresh = int(protocol['black-white-ratio (%)']/100.*self.ndots)

        # print(iThresh, self.ndots)

        np.random.seed(int(protocol['seed (#)']))

        index = 0
        for i in range(self.ndots):
            self.experiment['dot-x-%i' % i] = \
                        np.random.uniform(self.x.min()+\
                            self.experiment['size'][index]/2.,
                                          self.x.max()-\
                            self.experiment['size'][index]/2.,
                            len(self.experiment['index']))
            self.experiment['dot-z-%i' % i] = \
                        np.random.uniform(self.z.min()+\
                            self.experiment['size'][index]/2.,
                                          self.z.max()-\
                            self.experiment['size'][index]/2.,
                            len(self.experiment['index']))
            if i>=iThresh:
                self.experiment['dot-color-%i' % i] = 0.*\
                        np.ones(len(self.experiment['index']))
            else:
                self.experiment['dot-color-%i' % i] = 1.*\
                        np.ones(len(self.experiment['index']))


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        img = init_bg_image(self, index)

        for i in range(self.ndots):

            self.add_dot(img, 
                        (self.experiment['dot-x-%i' % i][index],
                         self.experiment['dot-z-%i' % i][index]),
                         self.experiment['size'][index],
                         self.experiment['dot-color-%i' % i][index])

        return img

def insure_no_overlap_with():
    pass

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('sparse-noise')
    params['N-repeat'] = 10

    import time
    import cv2 as cv

    Stim = stim(params)

    t0, index = time.time(), 0
    while True and index<(len(Stim.experiment['index'])-1):
        index = int((time.time()-t0)\
                /Stim.protocol['presentation-duration'])
        cv.imshow("Video Output", 
                  Stim.get_image(index).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
