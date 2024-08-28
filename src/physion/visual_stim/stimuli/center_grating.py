import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

####################################
##  ----    CENTER GRATING --- #####
####################################

  # default param values:
params = {"presentation-duration":3,
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "angle (deg)":90,
          "radius (deg)":220,
          "phase (deg)":0.,
          "spatial-freq (cycle/deg)":0.04,
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['bg-color',
                               'x-center', 'y-center',
                               'radius','spatial-freq',
                               'angle', 'phase', 'contrast'])


    def get_image(self, episode, time_from_episode_start=0):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
                       angle=self.experiment['angle'][episode],
                       radius=self.experiment['radius'][episode],
                       phase_shift_Deg=self.experiment['phase'][episode]\
                               if 'phase' in self.experiment else 90.,
                       spatial_freq=self.experiment['spatial-freq'][episode],
                       contrast=self.experiment['contrast'][episode],
                       xcenter=self.experiment['x-center'][episode],
                       zcenter=self.experiment['y-center'][episode])
        return img



if __name__=='__main__':

    import physion.utils.plot_tools as pt
    from physion.visual_stim.build import get_default_params

    params = get_default_params('center-grating')

    params['units'] = 'deg'
    Stim = stim(params)
    params['units'] = 'cm'
    Stim2 = stim(params)

    fig, AX = pt.figure(axes=(2,1), 
            figsize=(1.8,2), wspace=0, left=0, right=0, bottom=0.1, top=0.5)

    AX[0].set_title('angular space')
    AX[1].set_title('on screen')

    Stim.plot_stim_picture(0, ax=AX[0], with_mask=True)
    Stim2.plot_stim_picture(0, ax=AX[1])

    pt.plt.show()
