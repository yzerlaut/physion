"""
==========================================
 ---  template for new visual stimuli ---
==========================================

copy this and rename to the desired script name

[!!] need to add the new script to the "stimuli/__init__.py" 
"""
import numpy as np

from physion.visual_stim.main import visual_stim, init_bg_image

##########################################
##  ----    STIMULUS TEMPLATE    --- #####
##########################################

params = {\
  "presentation-duration": 2.0,
  # ------------------------------------
  # general properties
  "N_repeat": 5,
  "N_deviant": 1.0,
  "N_redundant": 7.0,
  "Nmin-successive-redundant": 3.0,
  "stim-duration": 2.0,
  "seed": 1.0,
  "stimulus": "grating",
  # ------------------------------------
  # case: "grating"
  "angle": 0,
  "angle-redundant": 45.0,
  "angle-deviant": 135.0,
  "spatial-freq": 0.04,
  "contrast": 1.0,
  # ----------------------------------- 
  "bg-color":0.5
}   

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        # fix pseudo-random seed
        np.random.seed(int(protocol['seed']))

        # initialize a default sequence:
        NperRepeat = int(protocol['N_deviant']+protocol['N_redundant'])
        params['N-repeat'] = NperRepeat*protocol['N_repeat']

        super().__init__(protocol, params)


        # that we modify to stick to the oddball protocol
        N = int(NperRepeat-protocol['Nmin-successive-redundant'])
        iShifts = np.random.randint(0, N, protocol['N_repeat'])

        NmSR = int(protocol['Nmin-successive-redundant'])
        for repeat in range(protocol['N_repeat']):

            # first redundants until random iShift
            for i in range(repeat*NperRepeat,
                           repeat*NperRepeat+NmSR+iShifts[repeat]):
                self.experiment['angle'][i] = params['angle-redundant']
            # deviant at random iShift
            self.experiment['angle'][repeat*NperRepeat+NmSR+iShifts[repeat]] = \
                                params['angle-deviant']

            # last redundants from random iShift
            for i in range(repeat*NperRepeat+NmSR+iShifts[repeat]+1,
                           (repeat+1)*NperRepeat):
                self.experiment['angle'][i]  = params['angle-redundant']

        self.refresh_freq = protocol['movie_refresh_freq']

    def get_image(self, index,
                  time_from_episode_start=0,
                  parent=None):
        """ 
        return the frame at a given time point
        """
        img = init_bg_image(self, index)
        self.add_grating_patch(img,
               angle=self.experiment['angle'][index],
               radius=200.,
               spatial_freq=self.experiment['spatial-freq'][index],
               contrast=self.experiment['contrast'][index])

        return img


if __name__=='__main__':

    from physion.visual_stim.build import get_default_params

    params = get_default_params('oddball')
    params['N-repeat'] = 3
    params['size'] = 0.1

    import time
    import cv2 as cv

    Stim = stim(params)

    t0 = time.time()
    while True:
        for i in range(10):
            cv.imshow("Video Output", Stim.get_image(i))
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
