import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

##############################################
##  ----    CENTER DRIFTING GRATINGS --- #####
##############################################

params = {"movie_refresh_freq":10,
          "presentation-duration":3,
          # default param values:
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "size (deg)":4.,
          "angle (deg)":0,
          "radius (deg)":40.,
          "speed (cycle/s)":1,
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
                         keys=['bg-color', 'speed',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'contrast'])

        ## /!\ inside here always use self.refresh_freq 
        ##        not the parent cls.refresh_freq 
        # when the parent multiprotocol will have ~10Hz refresh rate,
        ##                this can remain 2-3Hz
        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, episode, time_from_episode_start=0):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
                       angle=self.experiment['angle'][episode],
                       radius=self.experiment['radius'][episode],
                       spatial_freq=self.experiment['spatial-freq'][episode],
                       contrast=self.experiment['contrast'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode],
                       time_phase=self.experiment['speed'][episode]*time_from_episode_start)
        return img

    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, vse=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        ax = self.show_frame(episode, ax=ax, label=label,
                             parent=parent)
        arrow['direction'] = self.experiment['angle'][episode]
        arrow['center'] = [self.experiment['x-center'][episode],
                           self.experiment['y-center'][episode]]
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
