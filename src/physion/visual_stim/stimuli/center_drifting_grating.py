import numpy as np

from physion.visual_stim.main import vis_stim_image_built,\
        init_times_frames, init_bg_image

##############################################
##  ----    CENTER DRIFTING GRATINGS --- #####
##############################################

params = {"movie_refresh_freq":10,
          # default param values:
          "presentation-duration":3,
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "size (deg)":4.,
          "angle (deg)":0,
          "radius (deg)":40.,
          "speed (cycle/s)":1,
          "spatial-freq (cycle/deg)":0.04,
          "contrast":1.0,
          "bg-color (lum.)":0.5,
          # now we set the range of possible values:
          "x-center-1": -200, "x-center-2": 200, "N-x-center": 0,
          "y-center-1": -200, "y-center-2": 200, "N-y-center": 0,
          "size-1": 0, "size-2": 200, "N-size": 0,
          "angle-1": 0, "angle-2": 200, "N-angle": 0,
          "radius-1": 0, "radius-2": 200, "N-radius": 0,
          "speed-1": 0, "speed-2": 200, "N-speed": 0,
          "spatial-freq-1": 0, "spatial-freq-2": 200, "N-spatial-freq": 0,
          "contrast-1": 0, "contrast-2": 1, "N-contrast": 0,
          "bg-color-1": 0., "bg-color-2": 1., "N-bg-color": 0}
    

class stim(vis_stim_image_built):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color', 'speed',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'contrast'])

        ## /!\ inside here always use self.refresh_freq 
        ##        not the parent cls.refresh_freq 
        # when the parent multiprotocol will have ~10Hz refresh rate,
        ##                this can remain 2-3Hz
        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        self.add_grating_patch(img,
                       angle=cls.experiment['angle'][episode],
                       radius=cls.experiment['radius'][episode],
                       spatial_freq=cls.experiment['spatial-freq'][episode],
                       contrast=cls.experiment['contrast'][episode],
                       xcenter=cls.experiment['x-center'][episode],
                       zcenter=cls.experiment['y-center'][episode],
                       time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)
        arrow['direction'] = cls.experiment['direction'][episode]
        arrow['center'] = [cls.experiment['x-center'][episode],
                           cls.experiment['y-center'][episode]]
        self.add_arrow(arrow, ax)
        return ax

    ### HERE YOU CAN OVERWRITE THE DEFAULT plot_stim_picture FUNCTION

    # def plot_stim_picture(self, episode, ax,
                          # parent=None, 
                          # label={'degree':20,
                                 # 'shift_factor':0.02,
                                 # 'lw':1, 'fontsize':10},
                          # vse=False,
                          # arrow={'length':20,
                                 # 'width_factor':0.05,
                                 # 'color':'red'}):

        # """
        # """
        # cls = (parent if parent is not None else self)

        # tcenter = .5*(cls.experiment['time_stop'][episode]-\
                      # cls.experiment['time_start'][episode])
        
        # ax = self.show_frame(episode, tcenter, ax=ax,
                             # parent=parent,
                             # label=label)
