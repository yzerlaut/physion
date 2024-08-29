import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    RANDOM DOTS    --- #####
####################################

params = {"movie_refresh_freq":30.,
          "presentation-duration":3,
          "refresh (Hz)":1.,
          "size (deg)":5.,
          "ndots (#)":7,
          "seed (#)":1,
          "dotcolor (lum.)":-1,
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

    params = get_default_params('random-dots')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
