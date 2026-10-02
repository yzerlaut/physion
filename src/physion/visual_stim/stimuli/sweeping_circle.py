import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
###  ----    SWEEPING CIRCLE    ----  ####
##########################################

params = {\
      # default param values:
      "presentation-duration":4, # should be: sweeping-start + sweeping-duration
      "sweeping-duration":4, # second
      "sweeping-start":0., # second
      "radius":5., #degre
      "color":0.,
      "contrast":1,
      "trajectory": "bottom-left_to_top-right",
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

    def get_image(self, episode_index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """

        img = init_bg_image(self, episode_index)

        # Define position of circle
        tfrac = np.clip((time_from_episode_start-self.experiment['sweeping-start'][episode_index])/self.experiment['sweeping-duration'][episode_index], 0, 1)

        if self.experiment['trajectory'][episode_index] == "bottom-left_to_top-right":
            x0 = self.x[0,0] + tfrac*(self.x[-1,-1]-self.x[0,0])
            z0 = self.z[0,0] + tfrac*(self.z[-1,-1]-self.z[0,0])
        elif self.experiment['trajectory'][episode_index] == "top-left_to_bottom-right":
            x0 = self.x[0,0] + tfrac*(self.x[-1,-1]-self.x[0,0])
            z0 = self.z[-1,-1] + tfrac*(self.z[0,0]-self.z[-1,-1])
        elif self.experiment['trajectory'][episode_index] == "top-right_to_bottom-left":
            x0 = self.x[-1,-1] + tfrac*(self.x[0,0]-self.x[-1,-1])
            z0 = self.z[-1,-1] + tfrac*(self.z[0,0]-self.z[-1,-1])
        elif self.experiment['trajectory'][episode_index] == "bottom-right_to_top-left":
            x0 = self.x[-1,-1] + tfrac*(self.x[0,0]-self.x[-1,-1])
            z0 = self.z[0,0] + tfrac*(self.z[-1,-1]-self.z[0,0])
        else :
            raise ValueError(f"trajectory {self.experiment['trajectory'][episode_index]} not implemented")

        # Define color of circle
        radius = self.experiment['radius'][episode_index]
        contrast = self.experiment['contrast'][episode_index] * (self.experiment['bg-color'][episode_index] - self.experiment['color'][episode_index])
        img[((self.x-x0)**2 + (self.z-z0)**2) < radius**2] = self.experiment['bg-color'][episode_index] - contrast

        """
        # do you image construction/processing here:
        for i in range(int(self.experiment['ndots'][index])):

            pos = np.random.randn(2)*self.experiment['radius'][index]

            self.add_dot(img, pos,
                         self.experiment['size'][index],
                         self.experiment['color'][index],
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

    params = get_default_params('sweeping-circle')
    """ params["color"]=1.
    params["contrast"]=0.5
    params["bg-color"]=0. """

    import time
    import cv2 as cv

    Stim = stim(params)
    
    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
