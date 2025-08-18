import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
####  ----    DIMMING CIRCLE    ----  ####
##########################################

params = {\
      # default param values:
      "presentation-duration":5,
      "dimming-duration":1,
      "radius":25.,
      "x-center":0., # degree
      "y-center":0., # degree
      "color": 0.,
      "contrast": 1,
      "dimming-nonlinearity":3, # degree
      "bg-color":0.5,
}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol, params)

        self.refresh_freq = protocol['movie_refresh_freq']

    def get_circle_color(self, index, t):
        t_frac = np.clip(t/self.experiment['dimming-duration'][index], 0, 1)
        dColor = self.experiment['contrast'][index] * (self.experiment['color'][index] - self.experiment['bg-color'][index])

        nonlinearity = self.experiment['dimming-nonlinearity'][index]

        return self.experiment['bg-color'][index] + t_frac**nonlinearity * dColor
    
    def get_image(self, episode_index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """

        img = init_bg_image(self, episode_index)

        # circle patch with dimming color
        x0 = self.experiment['x-center'][episode_index]
        z0 = self.experiment['y-center'][episode_index]
        radius = self.experiment['radius'][episode_index]
        img[((self.x - x0)**2 + (self.z - z0)**2) < radius**2] = self.get_circle_color(episode_index, time_from_episode_start)

        """
        # do you image construction/processing here:
        for i in range(int(self.experiment['ndots'][index])):

            pos = np.random.randn(2)*self.experiment['radius'][index]

            self.add_dot(img, pos,
                         self.experiment['size'][index],
                         self.experiment['dotcolor'][index],
                         type='square')
        """

        return img


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
        # tcenter = .5*(self.experiment['time_stop'][episode]-\
                      # self.experiment['time_start'][episode])
        
        # ax = self.show_frame(episode, tcenter, ax=ax,
                             # parent=parent,
                             # label=label)

if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('dimming-circle')
    
    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
