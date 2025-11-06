import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

  # default param values:
params = {"presentation-duration":12,
          "flicker-size":10.,
          "flicker-freq":5.,
          "bar-size":5.,
          "direction":3, # 0-Up, 1-Down, 2-Left, 3-Right
          "contrast":1.0,
          "z-min":-47.0,
          "z-max":+47.0,
          "x-min":-63.0,
          "x-max":+63.0,
          "bg-color":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol, units='cm'):

        super().__init__(protocol, params)


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
            center = self.protocol['z-max']+T*(self.protocol['z-min']-self.protocol['z-max'])
        if self.experiment['direction'][episode]==1:
            center = self.protocol['z-min']+T*(self.protocol['z-max']-self.protocol['z-min'])
        if self.experiment['direction'][episode]==2:
            center = self.protocol['x-max']+T*(self.protocol['x-min']-self.protocol['x-max'])
        if self.experiment['direction'][episode]==3:
            center = self.protocol['x-min']+T*(self.protocol['x-max']-self.protocol['x-min'])


        # build the bar
        if self.experiment['direction'][episode] in [0,1]:
            bar_cond = (self.z<\
                        (center+self.experiment['bar-size'][episode]/2.)) &\
                    (self.z>\
                        (center-self.experiment['bar-size'][episode]/2.))
            DX = self.protocol['x-max']-self.protocol['x-min']
            flickSpace = np.arange(\
                    int(DX/self.experiment['bar-size'][episode])+2)*\
                                    self.experiment['bar-size'][episode]+\
                                            self.protocol['x-min']
            for i, f1, f2 in zip(range(len(flickSpace)-1),
                                 flickSpace[:-1], flickSpace[1:]):
                cond = bar_cond & (self.x>=f1) & (self.x<f2)
                img[cond] = (iFlicker+i%2)%2
            img[(self.x<self.protocol['x-min']) |\
                (self.x>self.protocol['x-max'])] = \
                            self.experiment['bg-color'][episode]

        if self.experiment['direction'][episode] in [2,3]:
            bar_cond = (self.x<\
                        (center+self.experiment['bar-size'][episode]/2.)) &\
                    (self.x>\
                        (center-self.experiment['bar-size'][episode]/2.))
            DZ = self.protocol['z-max']-self.protocol['z-min']
            flickSpace = np.arange(\
                    int(DZ/self.experiment['bar-size'][episode])+2)*\
                                    self.experiment['bar-size'][episode]+\
                                            self.z.min()
            for i, f1, f2 in zip(range(len(flickSpace)-1),
                                 flickSpace[:-1], flickSpace[1:]):
                cond = bar_cond & (self.z>=f1) & (self.z<f2)
                img[cond] = (iFlicker+i%2)%2
            img[(self.z<self.protocol['z-min']) |\
                (self.z>self.protocol['z-max'])] = \
                            self.experiment['bg-color'][episode]

        return img



if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('flickering_bar')
    params['x-min'] = -110
    params['x-max'] = 10
    params['direction'] = 2
    params['Screen'] = 'LN-VR-3screens'

    import time
    import cv2 as cv

    Stim = stim(params)
    print(Stim.z.min(), Stim.z.max())
    print(Stim.x.min(), Stim.x.max())

    t0 = time.time()
    while True:
        cv.imshow("Video Output", 
                  Stim.get_image(0, time_from_episode_start=time.time()-t0).T)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
