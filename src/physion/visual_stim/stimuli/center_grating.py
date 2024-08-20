import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

params = {"movie_refresh_freq":10,
          "presentation-duration":3,
          # stimulus parameters (add parenthesis with units):
          "x-center (deg)":-20.,
          "y-center (deg)":-20.,
          "angle (deg)":45,
          "radius (deg)":30,
          "spatial-freq (cycle/deg)":0.04,
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'contrast'])

        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, episode, time_from_episode_start=0):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
                       angle=self.experiment['angle'][episode],
                       radius=self.experiment['radius'][episode],
                       spatial_freq=self.experiment['spatial-freq'][episode],
                       contrast=self.experiment['contrast'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode])
        return img

    def plot_stim_picture(self, episode,
                          ax=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        return self.show_frame(episode, ax=ax, label=label)


if __name__=='__main__':

    import physion.utils.plot_tools as pt
    from physion.visual_stim.build import get_default_params

    params = get_default_params('center-grating')
    params['no-window'] = True

    params['demo'] = False
    Stim = stim(params)

    Stim.plot_stim_picture(0)
    pt.plt.show()
