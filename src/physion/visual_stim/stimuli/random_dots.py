import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

####################################
##  ----    RANDOM DOTS    --- #####
####################################

params = {"movie_refresh_freq":2,
          # default param values:
          "presentation-duration":3,
          "size (deg)":4.,
          "ndots (#)":7,
          "seed (#)":1,
          "dotcolor (lum.)":-1,
          "bg-color (lum.)":0.5}
    
def compute_new_image_with_dots(cls, index,
                                away_from_edges_factor=4):

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
                         keys=['bg-color', 'ndots', 'size', 'dotcolor', 'seed'])

        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        return compute_new_image_with_dots(self, index)

    def get_frames_sequence(self, index, parent=None):
        """
        get frame seq
        """

        time_indices, times, FRAMES = init_times_frames(self, index, self.refresh_freq)

        np.random.seed(int(self.experiment['seed'][index]+3*index)) # changing seed at each realization
        for iframe, t in enumerate(times):
            img = compute_new_image_with_dots(self, index)
            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, self.refresh_freq
