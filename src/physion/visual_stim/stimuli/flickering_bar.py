import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

  # default param values:
params = {"presentation-duration":12,
          "flicker-size (deg)":10.,
          "flicker-freq (Hz)":5.,
          "bar-size (deg)":5.,
          "direction (#)":3, # 0-Up, 1-Down, 2-Left, 3-Right
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol, units='cm'):

        super().__init__(protocol,
                         keys=['bar-size',
                               'flicker-size', 
                               'flicker-freq',
                               'bg-color',
                               'direction'])

    def get_image(self, episode, 
                  time_from_episode_start=0):
        """
        direction=0 --> bar going up
        direction=1 --> bar going down 
        direction=2 --> bar going left
        direction=3 --> bar going right
        """

        img = init_bg_image(self, episode)

        iFlicker = int(time_from_episode_start*\
                self.experiment['flicker-freq'][episode])%2

        T = (time_from_episode_start)/self.protocol['presentation-duration']

        if self.experiment['direction'][episode]==0:
            center = self.z.max()+T*(self.z.min()-self.z.max())
        if self.experiment['direction'][episode]==1:
            center = self.z.min()+T*(self.z.max()-self.z.min())
        if self.experiment['direction'][episode]==2:
            center = self.x.max()+T*(self.x.min()-self.x.max())
        if self.experiment['direction'][episode]==3:
            center = self.x.min()+T*(self.x.max()-self.x.min())


        if self.experiment['direction'][episode] in [0,1]:
            bar_cond = (self.z<\
                        (center+self.experiment['bar-size'][episode]/2.)) &\
                    (self.z>\
                        (center-self.experiment['bar-size'][episode]/2.))
            DX = self.x.max()-self.x.min()
            flickSpace = np.arange(\
                    int(DX/self.experiment['bar-size'][episode])+2)*\
                                    self.experiment['bar-size'][episode]+\
                                            self.x.min()
            for i, f1, f2 in zip(range(len(flickSpace)-1),
                                 flickSpace[:-1], flickSpace[1:]):
                cond = bar_cond & (self.x>=f1) & (self.x<f2)
                img[cond] = (iFlicker+i%2)%2

        if self.experiment['direction'][episode] in [2,3]:
            bar_cond = (self.x<\
                        (center+self.experiment['bar-size'][episode]/2.)) &\
                    (self.x>\
                        (center-self.experiment['bar-size'][episode]/2.))
            DZ = self.z.max()-self.z.min()
            flickSpace = np.arange(\
                    int(DZ/self.experiment['bar-size'][episode])+2)*\
                                    self.experiment['bar-size'][episode]+\
                                            self.z.min()
            for i, f1, f2 in zip(range(len(flickSpace)-1),
                                 flickSpace[:-1], flickSpace[1:]):
                cond = bar_cond & (self.z>=f1) & (self.z<f2)
                img[cond] = (iFlicker+i%2)%2

        return img



if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('flickering_bar')

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
