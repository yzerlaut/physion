import numpy as np

from physion.visual_stim.main import visual_stim,\
        init_times_frames, init_bg_image

##########################################
##  ----    LOOMING STIMULUS     --- #####
##########################################

params = {"movie_refresh_freq":10,
          "presentation-duration":3,
          # default param values:
          "radius-start (deg)":10,
          "radius-end (deg)":200.,
          "x-center (deg)":0.,
          "y-center (deg)":0.,
          "color (lum.)":-1,
          "looming-nonlinearity (a.u.)":2,
          "looming-duration (s)":3,
          "end-duration (s)":1,
          "bg-color (lum.)":0.5}
    

class stim(visual_stim):
    """
    stimulus specific visual stimulation object

    all functions should accept a "parent" argument that can be the 
    multiprotocol holding this protocol
    """

    def __init__(self, protocol):

        super().__init__(protocol,
                         keys=['radius-start', 'radius-end',
                               'x-center', 'y-center',
                               'color', 'looming-nonlinearity', 'looming-duration',
                               'end-duration', 'bg-color'])

        self.refresh_freq = protocol['movie_refresh_freq']

    def compute_looming_trajectory(self,
                                   duration=1., nonlinearity=2,
                                   dt=30e-3, start_size=0.5, end_size=100):
        """ SIMPLE """
        time_indices = np.arange(int(duration/dt))*dt
        angles = np.linspace(0, 1, len(time_indices))**nonlinearity
        return time_indices, angles*(end_size-start_size)+start_size

    def get_frames_sequence(self, index):
        """
        
        """
        # background frame:
        bg = 2*self.experiment['bg-color'][index]-1.+0.*self.x

        interval = self.experiment['time_stop'][index]-self.experiment['time_start'][index]
        t, angles = self.compute_looming_trajectory(duration=self.experiment['looming-duration'][index],
                                                    nonlinearity=self.experiment['looming-nonlinearity'][index],
                                                    dt=1./self.refresh_freq,
                                                    start_size=self.experiment['radius-start'][index],
                                                    end_size=self.experiment['radius-end'][index])

        itend = int(1.2*interval*self.refresh_freq)

        times_index_to_frames, FRAMES = [0], [bg.copy()]
        for it in range(len(t))[1:]:
            img = bg.copy()
            self.add_dot(img, (self.experiment['x-center'][index], self.experiment['y-center'][index]),
                         angles[it],
                         self.experiment['color'][index],
                         type='circle')
            FRAMES.append(img)
            times_index_to_frames.append(it)
        it = len(t)-1
        while it<len(t)+int(self.experiment['end-duration'][index]*self.refresh_freq):
            times_index_to_frames.append(len(FRAMES)-1) # the last one
            it+=1
        while it<itend:
            times_index_to_frames.append(0) # the first one (bg)
            it+=1

        return times_index_to_frames, FRAMES, self.refresh_freq

    def get_image(self, index, time_from_episode_start=0, parent=None):
        img = self.experiment['bg-color'][index]+0.*self.x
        self.add_dot(img, (self.experiment['x-center'][index],
                           self.experiment['y-center'][index]),
                     self.experiment['radius-end'][index]/4.,
                     self.experiment['color'][index],
                     type='circle')
        return img

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

