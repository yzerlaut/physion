import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    LOOMING STIMULUS     --- #####
##########################################

params = {"movie_refresh_freq":30,
          "presentation-duration":3, # should be: looming-duration + end-duration
          "looming-duration (s)":3,
          "end-duration (s)":1,
          "radius-start (deg)":5,
          "radius-end (deg)":110.,
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "color (lum.)":-1,
          "looming-nonlinearity (a.u.)":2,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['radius-start', 
                               'radius-end',
                               'x-center', 'y-center',
                               'color', 
                               'looming-nonlinearity', 
                               'looming-duration',
                               'end-duration', 
                               'bg-color'])


    def get_circle_size(self, index, t):
        t_frac = np.clip(t/self.experiment['looming-duration'][index], 0, 1)

        start_size=self.experiment['radius-start'][index]
        end_size=self.experiment['radius-end'][index]
        dSize = end_size-start_size

        nonlinearity=self.experiment['looming-nonlinearity'][index]

        return start_size + t_frac**nonlinearity * dSize

    def get_image(self, index, time_from_episode_start=0, parent=None):
        img = self.experiment['bg-color'][index]+0.*self.x

        self.add_dot(img, (self.experiment['x-center'][index], self.experiment['y-center'][index]),
                     self.get_circle_size(index, time_from_episode_start),
                     self.experiment['color'][index],
                     type='circle')
        return img

    """
    def plot_stim_picture(self, episode,
                          ax=None, parent=None, label=None, enhance=False,
                          arrow={'length':10,
                                 'width_factor':0.05,
                                 'color':'red'}):

        ax = self.show_frame(episode, ax=ax, label=label, enhance=enhance,
                             parent=parent)

        l = self.experiment['radius-end'][episode]/3.8 # just like above
        for d in np.linspace(0, 2*np.pi, 3, endpoint=False):
            arrow['center'] = [self.experiment['x-center'][episode]+np.cos(d)*l+\
                               np.cos(d)*arrow['length']/2.,
                               self.experiment['y-center'][episode]+np.sin(d)*l+\
                               np.sin(d)*arrow['length']/2.]
                
            arrow['direction'] = -180*d/np.pi
            self.add_arrow(arrow, ax)
            
        return ax
    """

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('looming-stim')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
