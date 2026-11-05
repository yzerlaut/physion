import types
import numpy as np

# legacy protocols: saccade amplitude in screen pixels ("saccade-amplitude"),
#   200 pixels when not specified
LEGACY_SACCADE_AMPLITUDE = 200.

# parameters added to a stimulus by its "-VSE" version
params = {
      "min-saccade-duration":0.1,
      "max-saccade-duration":1.0,
      "saccade-amplitude-deg":30.0, # in degrees (legacy protocols:
                                    #   "saccade-amplitude" in screen pixels)
      "seed":0
}


def with_vse(base):
    """
    "-VSE" version of a stimulus: its content moves with the VSE, behind a fixed aperture
        ("x-center", "y-center", "radius")

    base: module of a stimulus whose `stim` class has a
            get_content_image(index, time_from_episode_start=0, shift=(0, 0)) method
            (image before aperture, content shifted by `shift` in degrees), and that
            can define a `legacy_vse_deg_per_pixel` attribute (degrees of a shift in pixels,
            default: the angle per pixel at the center of the screen)

    returns an object like a stimulus module: with `params` and `stim`
    """
    if not hasattr(base.stim, 'get_content_image'):
        raise AttributeError(' [!!] stimulus "%s" has no "-VSE" version: it needs a "get_content_image" method [!!]' %\
                                base.__name__.split('.')[-1])

    # no second grating in the VSE: only the content within the aperture is shifted
    vse_params = {**{key:val for key, val in base.params.items() if not key.endswith('-surround')},
                  **params}

    class stim(base.stim):

        def __init__(self, protocol):

            super().__init__(protocol, vse_params)
            self.vse = generate_sequence_from_protocol(protocol)

        def get_vse_shift(self, time_from_episode_start=0):
            """ (x, y) shift of the VSE at that time, in degrees """
            dx, dy = compute_shifts(self, time_from_episode_start)
            if self.vse.get('units', 'pixels')=='pixels':
                # legacy protocols: shifts in screen pixels
                ddx, ddz = self.get_legacy_vse_deg_per_pixel()
                dx, dy = dx*ddx, dy*ddz
            return (float(dx), float(dy))

        def get_legacy_vse_deg_per_pixel(self):
            if hasattr(self, 'legacy_vse_deg_per_pixel'):
                return self.legacy_vse_deg_per_pixel
            from physion.visual_stim.geometry import deg_per_pixel
            return deg_per_pixel(self.screen, self.units)

        def get_image(self, index,
                      time_from_episode_start=0,
                      parent=None):

            img = self.get_content_image(index,
                        time_from_episode_start=time_from_episode_start,
                        shift=self.get_vse_shift(time_from_episode_start))

            return self.blank_surround(img,
                            bg_color=self.experiment['bg-color'][index],
                              xcenter=self.experiment['x-center'][index],
                              zcenter=self.experiment['y-center'][index],
                              radius = self.experiment['radius'][index])

    return types.SimpleNamespace(params=vse_params, stim=stim)


def generate_sequence(duration=2,
                        min_saccade_duration=0.5,# in s
                        max_saccade_duration=2.,# in s
                        saccade_amplitude=200, # in "units"
                        units='pixels', # 'deg' or 'pixels' (legacy)
                        seed=0,
                        presentation_duration=None, # in s, saccades until its end
                        shift_seed=None,
                        verbose=False):
    """
    A simple 'Virtual-Saccadic-Exploration' (VSE) generator
    based on temporal and spatial shifts drown form a uniform random distribution

    shifts in visual degrees (units='deg') or in screen pixels (units='pixels', legacy)

    min_saccade_duration=max_saccade_duration: periodic saccades

    presentation_duration: saccades are added after the first ones until that time
                                (None: the sequence can end before the presentation)
    shift_seed: shifts independent of the saccade times: the same (x, y) series for
                                the same shift_seed and amplitude, whatever the timing
    """

    if (min_saccade_duration<0) or (max_saccade_duration<=0) or\
            (max_saccade_duration<min_saccade_duration):
        raise ValueError(' [!!] we need 0<=min-saccade-duration<=max-saccade-duration and max-saccade-duration>0 [!!]')

    if verbose:
        print('generating Virtual-Scene-Exploration (with seed "%s") [...]' % seed)

    np.random.seed(seed)

    if max_saccade_duration>min_saccade_duration:
        nsaccades = int(3*duration/(max_saccade_duration-min_saccade_duration))
    else:
        nsaccades = int(np.ceil(duration/max_saccade_duration))

    tsaccades = np.cumsum(np.random.uniform(min_saccade_duration, max_saccade_duration,
                                            size=nsaccades))

    x = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))
    y = np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude, size=len(tsaccades))

    if presentation_duration is not None:
        while (len(tsaccades)==0) or (tsaccades[-1]<presentation_duration):
            t0 = tsaccades[-1] if len(tsaccades)>0 else 0.
            tsaccades = np.append(tsaccades,
                                  t0+np.random.uniform(min_saccade_duration, max_saccade_duration))
            x = np.append(x, np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude))
            y = np.append(y, np.random.uniform(saccade_amplitude/5., 2*saccade_amplitude))

    if shift_seed is not None:
        # (x, y) pairs: the first shifts do not depend on the number of saccades
        x, y = np.random.RandomState(shift_seed).uniform(saccade_amplitude/5., 2*saccade_amplitude,
                                                         size=(len(tsaccades), 2)).T

    dtype = int if units=='pixels' else float
    return {'t':np.array(list(tsaccades)),
            'x':np.array(list(x), dtype=dtype),
            'y':np.array(list(y), dtype=dtype),
            'max_amplitude':saccade_amplitude,
            'units':units}


def generate_jittered_sequence(regular_saccade_rate, # in Hz
                               jitter_frames, # offset of each saccade, in movie frames
                               refresh_rate, # in Hz
                               duration, # in s
                               saccade_amplitude=30., # in degrees
                               seed=0,
                               shift_seed=None):
    """
    VSE on a regular grid of saccades, each one moved earlier or later by `jitter_frames`:

        t_k = k/regular_saccade_rate +/- jitter_frames/refresh_rate

    the +/- pattern (from `seed`) and the shifts do not depend on `jitter_frames`, so that:
        - jitter_frames=0 is the regular sequence
        - the sequences of a given seed are spaced by exactly |jitter_1-jitter_2| frames
    """
    frames_per_saccade = refresh_rate/regular_saccade_rate
    if abs(frames_per_saccade-round(frames_per_saccade))>1e-9:
        print(' [!!] the refresh rate (%.1f Hz) is not a multiple of "regular-saccade-rate" (%.2f Hz):' %\
                    (refresh_rate, regular_saccade_rate))
        print('      the regular saccades are moved to the closest frame [!!]')

    if jitter_frames!=int(jitter_frames):
        raise ValueError(' [!!] "saccade-jitter-frames" should be a whole number of frames [!!]')

    nsaccades = int(np.ceil(duration*regular_saccade_rate-1e-9))-1
    grid = np.round(np.arange(1, nsaccades+1)*frames_per_saccade)

    if 2*jitter_frames>=np.floor(frames_per_saccade):
        raise ValueError(' [!!] the jitter of %i frames can change the order of the saccades: we need "saccade-jitter-frames"<%.1f [!!]' %\
                            (jitter_frames, np.floor(frames_per_saccade)/2.))

    rng = np.random.RandomState(seed)
    # as many saccades moved earlier and later, to keep the mean rate
    signs = rng.permutation(np.where(np.arange(nsaccades)%2==0, 1, -1))

    # (x, y) pairs: the first shifts do not depend on the number of saccades
    x, y = (rng if shift_seed is None else np.random.RandomState(shift_seed)).uniform(\
                saccade_amplitude/5., 2*saccade_amplitude, size=(nsaccades, 2)).T

    return {'t':(grid+jitter_frames*signs)/refresh_rate,
            'x':x,
            'y':y,
            'max_amplitude':saccade_amplitude,
            'units':'deg'}


def generate_sequence_from_protocol(protocol):
    """
    VSE of a protocol:
        - "regular-saccade-rate" in the protocol: regular saccades,
                moved by "saccade-jitter-frames" (see `generate_jittered_sequence`)
        - "saccade-amplitude-deg" in the protocol: shifts in visual degrees,
                saccades until the end of the presentation
        - otherwise (legacy protocols): shifts in screen pixels ("saccade-amplitude"),
                kept as is to rebuild them exactly
        - "saccade-shift-seed" (optional): same shifts whatever the saccade times
    """
    if 'regular-saccade-rate' in protocol:
        return generate_jittered_sequence(protocol['regular-saccade-rate'],
                                          protocol.get('saccade-jitter-frames', 0),
                                          protocol['movie_refresh_freq'],
                                          protocol['presentation-duration'],
                                          saccade_amplitude=protocol['saccade-amplitude-deg'],
                                          seed=protocol['seed'],
                                          shift_seed=protocol.get('saccade-shift-seed'))
    if 'saccade-amplitude-deg' in protocol:
        amplitude, units = protocol['saccade-amplitude-deg'], 'deg'
        presentation_duration = protocol.get('presentation-duration')
    else:
        amplitude = protocol.get('saccade-amplitude', LEGACY_SACCADE_AMPLITUDE)
        units = 'pixels'
        presentation_duration = None
    return generate_sequence(seed=protocol['seed'],
                             min_saccade_duration=protocol['min-saccade-duration'],
                             max_saccade_duration=protocol['max-saccade-duration'],
                             saccade_amplitude=amplitude,
                             units=units,
                             presentation_duration=presentation_duration,
                             shift_seed=protocol.get('saccade-shift-seed'))


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
    # tolerance: a saccade on a frame is displayed on that frame, despite rounding errors
    iS = np.flatnonzero(time_from_episode_start+1e-9>=self.vse['t'])
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


if __name__=='__main__':

    # demo: python -m physion.visual_stim.stimuli.virtual_scene_exploration grating-VSE
    import sys, time
    import cv2 as cv
    from physion.visual_stim.build import build_stim, get_default_params

    params = get_default_params(sys.argv[1] if len(sys.argv)>1 else 'images-VSE')
    print(params)

    Stim = build_stim(params)

    t0 = time.time()
    while True:
        # top row first, as in the movie frames (cv.imshow: row 0 at the top)
        cv.imshow("Video Output",
                  np.rot90(Stim.get_image(0, time_from_episode_start=time.time()-t0), k=1))
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
