import numpy as np

from physion.visual_stim.main import vis_stim_image_built,\
        init_times_frames, init_bg_image

###################################################
##  ----    oscillatory DRIFTING GRATINGS --- #####
###################################################

params = {"movie_refresh_freq":20,
          "presentation-duration":1,
          # default param values:
          "frequency (Hz)":3,
          "speed (cycle/s)":1,
          "angle (deg)":0,
          "spatial-freq (cycle/deg)":0.04,
          "contrast (lum.)":1.0,
          "bg-color (lum.)":0.5}
    

class stim(vis_stim_image_built):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):
        
        super().__init__(protocol,
                         keys=['bg-color', 'frequency', 'speed',
                               'angle', 'spatial-freq', 'contrast'])

        # WE REBUILD THE TIME COURSE OF THE EXPERIMENT
        for i in range(len(self.experiment['time_start'])):
            freq = self.experiment['frequency'][i]
            duration = max([4, 5/freq]) # maximum between 4s and 5 cycles
            print(duration)
            if i>0:
                self.experiment['time_start'][i] = self.experiment['time_stop'][i-1]+self.experiment['interstim'][i-1]
            self.experiment['time_duration'][i] = duration
            self.experiment['time_stop'][i] = self.experiment['time_start'][i]+duration

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        cls = (parent if parent is not None else self)
        img = init_bg_image(cls, episode)
        self.add_grating_patch(img,
                       angle=cls.experiment['angle'][episode],
                       radius=200,
                       spatial_freq=cls.experiment['spatial-freq'][episode],
                       contrast=cls.experiment['contrast'][episode]*np.sin(2*np.pi*time_from_episode_start*cls.experiment['frequency'][episode]),
                       xcenter=0, zcenter=0,
                       time_phase=cls.experiment['speed'][episode]*time_from_episode_start)
        return img


    def get_frames_sequence(self, index, parent=None):
        """
        we build a sequence of frames by successive calls to "self.get_image" 

        here we use self.refresh_freq, not cls.refresh_freq
         """
        cls = (parent if parent is not None else self)

        # we adapt the refresh freq accoring to the stim freq
        refresh_freq = cls.experiment['frequency'][index]*20.

        time_indices, times, FRAMES = init_times_frames(cls, index,\
                                                        refresh_freq)

        for iframe, t in enumerate(times):
            img = self.get_image(index, t,
                                 parent=parent) 
            FRAMES.append(self.image_to_frame(img))

        return time_indices, FRAMES, refresh_freq

