import numpy as np

def generate_sequence(duration=2,
                        min_saccade_duration=0.5,# in s
                        max_saccade_duration=2.,# in s
                        saccade_amplitude=200, # in pixels, TO BE PUT IN DEGREES
                        seed=0,
                        verbose=False):
    """
    A simple 'Virtual-Saccadic-Exploration' (VSE) generator
    based on temporal and spatial shifts drown form a uniform random distribution
    """
    
    if verbose:
        print('generating Virtual-Scene-Exploration (with seed "%s") [...]' % seed)
    
    np.random.seed(seed)
    
    tsaccades = np.cumsum(np.random.uniform(min_saccade_duration, max_saccade_duration,
                                            size=int(3*duration/(max_saccade_duration-min_saccade_duration))))

    x = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    
    return {'t':np.array(list(tsaccades)),
            'x':np.array(list(x), dtype=int),
            'y':np.array(list(y), dtype=int),
            'max_amplitude':saccade_amplitude}


def compute_shifted_pixels(self,
                  time_from_episode_start=0):

    iS = np.flatnonzero(time_from_episode_start>=self.vse['t'])
    if len(iS)>0:
        ix, iy = self.vse['x'][iS[-1]], self.vse['y'][iS[-1]]
    else:
        ix, iy = 0, 0

    return ix, iy

def compute_shifted_image(self, img, 
                          time_from_episode_start=0):
    """

    assumes a self.vse 
    
    """
    sx, sy = img.shape
    ix, iy = compute_shifted_pixels(self, time_from_episode_start)

    new_im = np.zeros(img.shape)
    new_im[ix:,iy:] = img[:sx-ix,:sy-iy]
    new_im[:ix,:] = img[sx-ix:,:]
    new_im[:,:iy] = img[:,sy-iy:]
    new_im[:ix,:iy] = img[sx-ix:,sy-iy:]

    return new_im