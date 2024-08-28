import sys, pathlib
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################################
##  ----    SCATTERED MOVING DOTS          --- #####
####################################################

params = {"movie_refresh_freq":30.,
          "presentation-duration":3,
          "speed (deg/s)":60.,
          "size (deg)":4.,
          "spacing (deg)":12.,
          "direction (deg)":270.,
          "ndots (#)":9,
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
        Y0 = np.zeros(ndots)-interval*dy_per_time/2.
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
                         keys=['speed', 'bg-color', 'ndots', 'spacing',
                               'direction', 'size', 'dotcolor', 'seed'])


    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """

        img = init_bg_image(self, index)

        line = np.arange(int(self.experiment['ndots'][index]))*\
                self.experiment['spacing'][index]

        X0, Y0, dx_per_time, dy_per_time =\
            get_starting_point_and_direction_mv_dots(line,
                self.experiment['time_stop'][index]-\
                        self.experiment['time_start'][index],
                        self.experiment['direction'][index],
                        self.experiment['speed'][index],
                        int(self.experiment['ndots'][index]))

        for x0, y0 in zip(X0, Y0):

            new_position = (x0+dx_per_time*time_from_episode_start,
                            y0+dy_per_time*time_from_episode_start)
            self.add_dot(img, new_position,
                         self.experiment['size'][index],
                         self.experiment['dotcolor'][index])

        return img


    def plot_stim_picture(self, episode, 
                          ax=None,
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
        
        ax = self.show_frame(episode, tcenter, ax=ax)

        direction = self.experiment['direction'][episode]
        arrow['direction'] = ((direction+180)%180)+180

        arrow['direction'] = self.experiment['direction'][episode]+180
        # print(arrow['direction'])

        for shift in [-.5, 0, .5]:

            arrow['center'] = [shift*np.sin(np.pi/180.*direction)*self.x.max()/3.,
                               shift*np.cos(np.pi/180.*direction)*self.x.max()/3.]

            self.add_arrow(arrow, ax)

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('line-moving-dots')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
