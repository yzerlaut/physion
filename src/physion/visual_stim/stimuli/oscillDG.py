import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

###################################################
##  ----    oscillatory DRIFTING GRATINGS --- #####
###################################################

params = {"movie_refresh_freq":20,
          "presentation-duration":1,
          # default param values:
          "frequency (Hz)":5,
          "speed (cycle/s)":1,
          "angle (deg)":0,
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
                         keys=['bg-color', 'frequency', 'speed',
                               'angle', 'spatial-freq', 'contrast'])

        # WE REBUILD THE TIME COURSE OF THE EXPERIMENT
        for i in range(len(self.experiment['time_start'])):
            freq = self.experiment['frequency'][i]
            duration = max([5, 5/freq]) # maximum between 5s and 5 cycles
            print(duration)
            if i>0:
                self.experiment['time_start'][i] = self.experiment['time_stop'][i-1]+self.experiment['interstim'][i-1]
            self.experiment['time_duration'][i] = duration
            self.experiment['time_stop'][i] = self.experiment['time_start'][i]+duration

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        img = init_bg_image(self, episode)
        self.add_grating_patch(img,
                       angle=self.experiment['angle'][episode],
                       radius=200,
                       spatial_freq=self.experiment['spatial-freq'][episode],
                       contrast=self.experiment['contrast'][episode]*(1-np.cos(2*np.pi*time_from_episode_start*self.experiment['frequency'][episode]))/2,
                       xcenter=0, zcenter=0,
                       time_phase=self.experiment['speed'][episode]*time_from_episode_start)
        return img


    def get_frames_sequence(self, index, parent=None):
        """
        we build a sequence of frames by successive calls to "self.get_image" 

         """

        # we adapt the refresh freq accoring to the stim freq
        refresh_freq = max([self.experiment['frequency'][index]*20,
                            self.experiment['speed'][index]*5])

        time_indices, times, FRAMES = init_times_frames(self, index,\
                                                        refresh_freq)

	# length of one cycle
        T = 1./self.experiment['frequency'][index]
        N = int(T*refresh_freq)

        for iframe, t in enumerate(times):
            i_period = int((t*refresh_freq)%N)
            time_indices[iframe] = i_period
            if t<=T:
                img = self.get_image(index, t,
                                     parent=parent) 
                FRAMES.append(self.image_to_frame(img))		

        return time_indices, FRAMES, refresh_freq

