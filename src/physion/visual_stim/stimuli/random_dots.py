import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    RANDOM DOTS    --- #####
####################################

params = {"movie_refresh_freq":30.,
          # default param values:
          "presentation-duration":3,
          "refresh (Hz)":2.,
          "size (deg)":4.,
          "ndots (#)":7,
          "seed (#)":1,
          "dotcolor (lum.)":-1,
          "bg-color (lum.)":0.5}
    
def compute_new_image_with_dots(cls, index,
                                seed=0,
                                away_from_edges_factor=4):

    np.random.seed(seed)
    dot_size_pix = int(np.round(cls.angle_to_pix(cls.experiment['size'][index]),0))
    Nx = int(cls.x.shape[0]/dot_size_pix)
    Nz = int(cls.x.shape[1]/dot_size_pix)

    img = init_bg_image(cls, index)
    for n in range(cls.experiment['ndots'][index]):
        ix, iz = (np.random.choice(np.arange(away_from_edges_factor, Nx-away_from_edges_factor)[::2],1, replace=False)[0],
                np.random.choice(np.arange(away_from_edges_factor,Nz-away_from_edges_factor)[::2],1, replace=False)[0])
        img[dot_size_pix*ix:dot_size_pix*(ix+1),
            dot_size_pix*iz:dot_size_pix*(iz+1)] = cls.experiment['dotcolor'][index]
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

        return compute_new_image_with_dots(self, index, seed=new_seed)

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
