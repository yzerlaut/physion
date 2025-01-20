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
##  ----    RANDOM DOTS    --- #####
####################################

params = {"movie_refresh_freq":30.,
          "presentation-duration":20*60.,
          "refresh (Hz)":2.,
          "size (deg)":5.,
          "ndots (#)":7,
          "seed (#)":1,
          "contrast (lum.)":1.,
          "contrast (lum.)":1.,
          "bg-color (lum.)":0.5}
    
def compute_new_image_with_dots(cls, index,
                                seed=0,
                                away_from_edges_factor=3):

    np.random.seed(seed)
    dot_size= int(cls.experiment['size'][index])
    Nx = int(cls.x.max()/dot_size)
    Nz = int(cls.z.max()/dot_size)

    # dot center positions
    xx, zz = np.meshgrid(np.arange(-Nx+away_from_edges_factor, 
                                   Nx+1-away_from_edges_factor)[::2],
                         np.arange(-Nz+away_from_edges_factor, 
                                   Nz+1-away_from_edges_factor)[::2],
                         indexing='ij')

    ii = np.random.choice(np.arange(len(xx.flatten())),
                         int(cls.experiment['ndots'][index]), replace=False)
    X = xx.flatten()[ii]
    Z = zz.flatten()[ii]

    img = init_bg_image(cls, index)
    for x, z in zip(X, Z):

        cls.add_dot(img, (x*dot_size, z*dot_size),
                    cls.experiment['size'][index],
                    cls.experiment['dotcolor'][index])

    return img

class stim(visual_stim):
    """
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 
                               'ndots', 
                               'size', 
                               'refresh', 
                               'dotcolor', 
                               'seed'])


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        new_seed = (1+self.experiment['seed'][index])**2+\
            int(time_from_episode_start*self.experiment['refresh'][index])

        return compute_new_image_with_dots(self, index, seed=int(new_seed))

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('sparse-noise')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
