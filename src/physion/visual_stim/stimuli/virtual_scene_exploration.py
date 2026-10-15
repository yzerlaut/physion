import numpy as np

# legacy protocols: saccade amplitude in screen pixels ("saccade-amplitude"),
#   200 pixels when not specified
LEGACY_SACCADE_AMPLITUDE = 200.

def generate_sequence(duration=2,
                        min_saccade_duration=0.5,# in s
                        max_saccade_duration=2.,# in s
                        saccade_amplitude=200, # in "units"
                        units='pixels', # 'deg' or 'pixels' (legacy)
                        seed=0,
                        verbose=False):
    """
    A simple 'Virtual-Saccadic-Exploration' (VSE) generator
    based on temporal and spatial shifts drown form a uniform random distribution

    shifts in visual degrees (units='deg') or in screen pixels (units='pixels', legacy)
    """

    if verbose:
        print('generating Virtual-Scene-Exploration (with seed "%s") [...]' % seed)

    np.random.seed(seed)

    tsaccades = np.cumsum(np.random.uniform(min_saccade_duration, max_saccade_duration,
                                            size=int(3*duration/(max_saccade_duration-min_saccade_duration))))

    x = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))

    dtype = int if units=='pixels' else float
    return {'t':np.array(list(tsaccades)),
            'x':np.array(list(x), dtype=dtype),
            'y':np.array(list(y), dtype=dtype),
            'max_amplitude':saccade_amplitude,
            'units':units}


def generate_sequence_from_protocol(protocol):
    """
    VSE of a protocol:
        - "saccade-amplitude-deg" in the protocol: shifts in visual degrees
        - otherwise (legacy protocols): shifts in screen pixels ("saccade-amplitude")
    """
    if 'saccade-amplitude-deg' in protocol:
        amplitude, units = protocol['saccade-amplitude-deg'], 'deg'
    else:
        amplitude = protocol.get('saccade-amplitude', LEGACY_SACCADE_AMPLITUDE)
        units = 'pixels'
    return generate_sequence(seed=protocol['seed'],
                             min_saccade_duration=protocol['min-saccade-duration'],
                             max_saccade_duration=protocol['max-saccade-duration'],
                             saccade_amplitude=amplitude,
                             units=units)


def deg_to_pixels(self, x, y):
    """
    shifts in degrees -> screen pixels, with the angle per pixel
        at the center of the screen (see visual_stim.geometry.deg_per_pixel)
    """
    if not hasattr(self, 'vse_deg_per_pixel'):
        from physion.visual_stim.geometry import deg_per_pixel
        self.vse_deg_per_pixel = deg_per_pixel(self.screen, self.units)
    dx, dz = self.vse_deg_per_pixel
    return int(np.round(x/dx)), int(np.round(y/dz))


def vse_in_pixels(self, vse):
    """ (x, y) shifts of the VSE in screen pixels """
    if vse.get('units', 'pixels')=='deg':
        xy = [deg_to_pixels(self, x, y) for x, y in zip(vse['x'], vse['y'])]
        return np.array([p[0] for p in xy]), np.array([p[1] for p in xy])
    return vse['x'], vse['y']


def compute_shifts(self,
                   time_from_episode_start=0):
    """ (x, y) shift at that time, in the units of the VSE """
    iS = np.flatnonzero(time_from_episode_start>=self.vse['t'])
    if len(iS)>0:
        return self.vse['x'][iS[-1]], self.vse['y'][iS[-1]]
    else:
        return 0, 0


def compute_shifted_pixels(self,
                  time_from_episode_start=0):
    """ (x, y) shift at that time, in screen pixels """
    ix, iy = compute_shifts(self, time_from_episode_start)
    if self.vse.get('units', 'pixels')=='deg':
        return deg_to_pixels(self, ix, iy)
    return ix, iy


def compute_shifted_image(self, img,
                          time_from_episode_start=0):
    """

    assumes a self.vse

    image shifted by the VSE, wrapping around the borders
    """
    ix, iy = compute_shifted_pixels(self, time_from_episode_start)
    if self.vse.get('units', 'pixels')=='deg':
        return np.roll(np.asarray(img, dtype=float), (ix, iy), axis=(0, 1))
    else:
        return legacy_shifted_image(img, ix, iy)


def legacy_shifted_image(img, ix, iy):
    """
    shift of the legacy protocols (pixel units), kept as is to rebuild them exactly:
        the strips wrapping around the borders are not shifted along the other axis,
        and shifts larger than the image fail
    """
    sx, sy = img.shape
    new_im = np.zeros(img.shape)
    new_im[ix:,iy:] = img[:sx-ix,:sy-iy]
    new_im[:ix,:] = img[sx-ix:,:]
    new_im[:,:iy] = img[:,sy-iy:]
    new_im[:ix,:iy] = img[sx-ix:,sy-iy:]
    return new_im
