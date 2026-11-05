from . import uniform_bg, looming_stim, grating,\
        line_moving_dots, scattered_moving_dots,\
    	random_dots, images, oscillDG,\
        gaussian_blob, flickering_bar, sparse_noise,\
        oddball, template, virtual_scene_exploration,\
        sweeping_circle, dimming_circle,\
        grating_stripe, flickering_grating

# backward compatibility:
from .legacy_images import natural_image, natural_image_VSE, set_of_images # read-only
gaussian_blobs=gaussian_blob
center_drifting_grating=grating
center_grating=grating


def __getattr__(name):
    """ "<stimulus>_VSE": <stimulus> with a virtual scene exploration """
    if name.endswith('_VSE') and (name[:-4] in globals()):
        globals()[name] = virtual_scene_exploration.with_vse(globals()[name[:-4]])
        return globals()[name]
    raise AttributeError('module %s has no attribute %s' % (__name__, name))
