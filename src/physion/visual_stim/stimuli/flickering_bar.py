import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

  # default param values:
params = {"presentation-duration":3,
          # stimulus parameters (add parenthesis with units):
          "flicker-size (deg)":10.,
          "flicker-freq (Hz)":10.,
          "bar-size (deg)":10.,
          "direction (deg)":0.,
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
                         keys=['bg-color',
                               'bar-size',
                               'flicker-size', 'flicker-freq',
                               'direction', 'contrast'],
                         units=units)


    def get_image(self, episode, 
                  time_from_episode_start=0):

        img = init_bg_image(self, episode)

        cond = (self.z<self.experiment['bar-size'][episode]/2.) &\
                    (self.z>-self.experiment['bar-size'][episode]/2.)
        img[cond] = self.experiment['contrast'][episode]

        return img



if __name__=='__main__':

    import physion.utils.plot_tools as pt
    from physion.visual_stim.build import get_default_params

    params = get_default_params('flickering-bar')
    params['no-window'] = True
    params['demo'] = False

    Stim = stim(params, units='deg')
    Stim2 = stim(params, units='cm')

    fig, AX = pt.figure(axes=(2,1), 
            figsize=(1.8,2), wspace=0, left=0, right=0, bottom=0.1, top=0.5)

    AX[0].set_title('angular space')
    AX[1].set_title('on screen')

    Stim.plot_stim_picture(0, ax=AX[0], with_mask=True)
    Stim2.plot_stim_picture(0, ax=AX[1])

    pt.plt.show()
