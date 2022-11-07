import sys, pathlib
import numpy as np

from physion.visual_stim.main import vis_stim_image_built, init_times_frames, init_bg_image

####################################################
##  ----    SCATTERED MOVING DOTS          --- #####
####################################################

params = {"movie_refresh_freq":5,
          # default param values:
          "presentation-duration":3,
          "speed (deg/s)":60.,
          "size (deg)":4.,
          "spacing (deg)":10.,
          "direction (deg)":270.,
          "ndots (#)":7,
          "dotcolor (lum.)":-1,
          "bg-color (lum.)":0.5,
          # now we set the range of possible values:
          "speed-1": 0.01, "speed-2": 200.0, "N-speed": 0,
          "size-1": 0.01, "size-2": 100, "N-size": 0,
          "spacing-1": 0.001, "spacing-2": 100, "N-spacing": 0,
          "direction-1": 0, "direction-2": 270, "N-direction": 0,
          "ndots-1": 1, "ndots-2": 1000, "N-ndots": 0,
          "bg-color-1": 0., "bg-color-2": 1., "N-bg-color": 0,
          "dotcolor-1": -1, "dotcolor-2": 1, "N-dotcolor": 0}
    

def get_starting_point_and_direction_mv_dots(line,
                                             interval,
                                             direction,
                                             speed,
                                             ndots):

    X0, Y0 = [], []

    if direction==0:

        # right -> left
        dx_per_time, dy_per_time = -speed, 0
        X0 = np.zeros(ndots)-interval*dx_per_time/2.
        Y0 = line-line.mean()

    elif direction==180:
        # left -> right
        dx_per_time, dy_per_time = speed, 0
        X0 = np.zeros(ndots)-interval*dx_per_time/2.
        Y0 = line-line.mean()

    elif direction==90:
        # top -> bottom
        dx_per_time, dy_per_time = 0, -speed
        Y0 = np.zeros(ndots)-interval*dy_per_time/2.
        X0 = line-line.mean()

    elif direction==270:

        # top -> bottom
        dx_per_time, dy_per_time = 0, speed
        Y0 = np.zeros(ndots)-interval*dy_per_time/2.
        X0 = line-line.mean()

    else:
        print('direction "%i" not implemented !' % direction)

    return X0, Y0, dx_per_time, dy_per_time



class stim(vis_stim_image_built):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['speed', 'bg-color', 'ndots', 'spacing',
                               'direction', 'size', 'dotcolor', 'seed'])

        ## /!\ here always use self.refresh_freq not the parent cls.refresh_freq ##
        # when the parent multiprotocol will have ~10Hz refresh rate, this can  remain 2-3Hz
        self.refresh_freq = protocol['movie_refresh_freq']


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        cls = (parent if parent is not None else self)

        img = init_bg_image(cls, index)

        line = np.arange(int(cls.experiment['ndots'][index]))*\
                cls.experiment['spacing'][index]

        X0, Y0, dx_per_time, dy_per_time =\
            get_starting_point_and_direction_mv_dots(line,
            cls.experiment['time_stop'][index]-\
                    cls.experiment['time_start'][index],
                    cls.experiment['direction'][index],
                    cls.experiment['speed'][index],
                    int(cls.experiment['ndots'][index]))

        for x0, y0 in zip(X0, Y0):

            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)

            self.add_dot(img, new_position,
                         cls.experiment['size'][index],
                         cls.experiment['dotcolor'][index])

        return img


    def plot_stim_picture(self, episode, ax,
                          parent=None, 
                          label=None,
                          vse=False,
                          arrow={'length':20,
                                 'width_factor':0.05,
                                 'color':'red'}):

        """
        """
        cls = (parent if parent is not None else self)

        tcenter = .45*(cls.experiment['time_stop'][episode]-\
                      cls.experiment['time_start'][episode])
        
        ax = self.show_frame(episode, tcenter, ax=ax,
                             parent=parent)

        direction = cls.experiment['direction'][episode]
        arrow['direction'] = ((direction+180)%180)+180

        arrow['direction'] = cls.experiment['direction'][episode]+180
        print(arrow['direction'])

        for shift in [-.5, 0, .5]:

            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*cls.x.max()/3.,
                               shift*np.cos(np.pi/180.*direction)*cls.x.max()/3.]

            self.add_arrow(arrow, ax)
