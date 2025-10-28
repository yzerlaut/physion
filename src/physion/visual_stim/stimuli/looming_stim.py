import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    LOOMING STIMULUS     --- #####
##########################################

params = {\
      "presentation-duration":3, # should be: looming-duration + end-duration
      "looming-duration":3,  #  second
      "end-duration":1, # s
      "radius-start":5, # degree
      "radius-end":110., # degree
      "x-center":0., # degree
      "y-center":0., # degree
      "color":0., # degree
      "looming-nonlinearity":2, # degree
      "bg-color":0.5,
      "contrast":1.0,
} 
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)


    def get_circle_size(self, index, t):
        t_frac = np.clip(t/self.experiment['looming-duration'][index], 0, 1)

        start_size=self.experiment['radius-start'][index]
        end_size=self.experiment['radius-end'][index]
        dSize = end_size-start_size

        nonlinearity=self.experiment['looming-nonlinearity'][index]

        return start_size + t_frac**nonlinearity * dSize

    def get_image(self, index, time_from_episode_start=0, parent=None):
        print('t=', time_from_episode_start)
        img = self.experiment['bg-color'][index]+0.*self.x

        dColor = self.experiment['contrast'][index] * (self.experiment['color'][index] - self.experiment['bg-color'][index])
        color = self.experiment['bg-color'][index] + dColor
        self.add_dot(img, (self.experiment['x-center'][index], self.experiment['y-center'][index]),
                     self.get_circle_size(index, time_from_episode_start),
                     color,
                     type='circle')
        return img

    def plot_stim_picture(self, episode, 
                          ax=None, label=None, vse=False,
                          with_scale=False,
                          arrow={'length':20,
                                 'width_factor':0.05,
                                 'color':'red'},
                          with_mask=False):

        """
        """
        tcenter = .15*(self.experiment['time_stop'][episode]-\
                      self.experiment['time_start'][episode])
        if with_scale:
            if self.units in ['cm', 'lin-deg']:
                label={'size':10/self.heights.max()*self.z.max(),
                       'label':'10cm ',
                       'shift_factor':0.02,
                       'lw':1, 'fontsize':10}
            else:
                label={'size':20,'label':'20$^o$  ',
                       'shift_factor':0.02,
                       'lw':1, 'fontsize':10}
        else:
            label=None

        ax = self.show_frame(episode, tcenter,
                             ax=ax,
                             label=label,
                             with_mask=with_mask)
        return ax

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
