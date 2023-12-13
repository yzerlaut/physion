import sys, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

####################################################
##  ----    SCATTERED MOVING DOTS          --- #####
####################################################

params = {"movie_refresh_freq":5,
          "presentation-duration":3,
          # default param values:
          "speed (deg/s)":60.,
          "size (deg)":4.,
          "spacing (deg)":10.,
          "direction (deg)":270.,
          "ndots (#)":7,
          "dotcolor (lum.)":-1,
          "bg-color (lum.)":0.5}
    

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
        # Y0 = np.zeros(ndots)-interval*dy_per_time/2.
        Y0 = interval*dy_per_time/2.*(.5*np.abs(np.random.randn(ndots))-1)
        X0 = line-line.mean()

    else:
        print('direction "%i" not implemented !' % direction)

    return X0, Y0, dx_per_time, dy_per_time



class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """
    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['speed', 'bg-color', 'ndots',
                               'spacing', 'direction', 'size',
                               'dotcolor', 'seed'])

        self.refresh_freq = protocol['movie_refresh_freq']

        # we initialize the trajectories

        self.X0, self.Y0 = {}, {}
        self.dx_per_time, self.dy_per_time = {}, {}

        for i, index in enumerate(self.experiment['index']):

            line = np.arange(int(self.experiment['ndots'][i]))*\
                                 self.experiment['spacing'][i]

            self.X0[str(index)], self.Y0[str(index)],\
                self.dx_per_time[str(index)], self.dy_per_time[str(index)] =\
                    get_starting_point_and_direction_mv_dots(line,
                            self.experiment['time_stop'][i]-\
                            self.experiment['time_start'][i],
                            self.experiment['direction'][i],
                            self.experiment['speed'][i],
                            int(self.experiment['ndots'][i]))



    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        img = init_bg_image(self, index)

        Index = str(self.experiment['index'][index])
        for x0, y0 in zip(self.X0[Index], self.Y0[Index]):

            new_position = (x0+self.dx_per_time[Index]*time_from_episode_start,
                            y0+self.dy_per_time[Index]*time_from_episode_start)

            self.add_dot(img, new_position,
                         self.experiment['size'][index],
                         self.experiment['dotcolor'][index])

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
        tcenter = .45*(self.experiment['time_stop'][episode]-\
                      self.experiment['time_start'][episode])
        
        ax = self.show_frame(episode, tcenter, ax=ax,
                             parent=parent)

        direction = self.experiment['direction'][episode]
        arrow['direction'] = ((direction+180)%180)+180

        arrow['direction'] = self.experiment['direction'][episode]+180
        print(arrow['direction'])

        for shift in [-.5, 0, .5]:

            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*self.x.max()/3.,
                               shift*np.cos(np.pi/180.*direction)*self.x.max()/3.]

            self.add_arrow(arrow, ax)
