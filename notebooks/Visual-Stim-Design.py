# %% [markdown]
# # Visual Stimulation

# %%
import sys, os, json
import numpy as np

sys.path += ['../src'] # add src code directory for physion
#sys.path.append(os.path.join(os.path.expanduser('~'), 'work', 'physion', 'src'))
import physion
import physion.utils.plot_tools as pt
from physion.visual_stim.build import get_default_params

# %% [markdown]
# ## Single Screen Configuration

# %% [markdown]
# ## Dealing with the transformation from angles to display on the screen
#
# We want to convert the angular space of animal vision to the coordinates on a flat screen.
#
# The angular coordinates of animal vision are $\theta_x$ and $\theta_z$ that are the relative angles with respect to the center of the visual field of coordinates ($\theta_x$=0, $\theta_z$=0). This means that $\theta_x \in [-\pi/2,\pi/2]$ and $\theta_z \in [-\pi/2,\pi/2]$ (vision covers only half of the 3d space).
#
# <img src="docs/visual_stim/coordinates.svg" width=260 height=260 />
#
# We start from the [spherical coordinates](https://en.wikipedia.org/wiki/Spherical_coordinate_system) with the physics convention: $\theta$ is the polar angle and $\phi$ is the azimuthal angle. 
#
# The eye is the (0,0,0) reference point, the screen is perpendicular to the (x,y) axis and is placed at a distance y=C.
#
# The link between Cartesian and spherical coordinates is the following:
#
# $$ x = r \, \sin(\theta) \, \cos(\phi) $$
#
# $$ y = r \, \sin(\theta) \, \sin(\phi) $$
#
# $$ z = r \, \cos(\theta) $$
#
# The relationship with our coordinates are: $\theta_z = \pi/2 - \theta$ and $\theta_x = \pi/2 - \phi$, so:
#
#
# $$ x = r \, \sin(\pi/2 - \theta_z) \, \cos(\pi/2 - \theta_x) $$
#
# $$ y = r \, \sin(\pi/2 - \theta_z) \, \sin(\pi/2 - \theta_x) $$
#
# $$ z = r \, \cos(\pi/2 - \theta_z) $$
#
# Using [trigonometric identities](https://en.wikipedia.org/wiki/List_of_trigonometric_identities), we get:
#
# $$ x = r \, \cos(\theta_z) \, \sin(\theta_x) $$
#
# $$ y = r \, \cos(\theta_z) \, \cos(\theta_x) $$
#
# $$ z = r \, \sin(\theta_z) $$
#
# %% [markdown]
# # Single Screen Setting 
# A single screen placed perpendicular to the 45 deg. axis of the mouse eye 
# 
# (i.e. the center of its visual field)

# %% [markdown]
# The screen corresponds to the coordinates: $$ y = C $$
#
# So this imposes the constraint:
#
# $$
# r = \frac{C}{ \cos(\theta_z) \, \cos(\theta_x) }
# $$
#
# So:
#
# $$ x = \frac{C}{ \cos(\theta_z) \, \cos(\theta_x)} \, \cos(\theta_z) \, \sin(\theta_x)  = C \, \tan(\theta_x) $$
#
# $$ z = \frac{C}{ \cos(\theta_z) \, \cos(\theta_x)} \, \sin(\theta_z)  = C \, \frac{\tan(\theta_z)}{\cos(\theta_x)} $$
#
# ------------------------
#
# ** This is implemented in the [visual_stim/main.py](../../visual_stim/main.py) in the function `set_angle_meshgrid_1screen` **

# %% [markdown]
# ## Illustrating the space wrapping from angle-to-screen

# %%
z = np.linspace(-5, 5, 20)
x = 16./9.*z
X, Z = np.meshgrid(x, z, indexing='ij')
Y = 0.*X+7

fig, AX = pt.figure(axes = (2,1), ax_scale=(2,3), wspace=0.5)
AX[0].plot(X.flatten(), Z.flatten(), '.', ms=1)
AX[0].axis('equal')
x = np.arctan(X/Y)
z = np.arctan(Z*np.cos(x)/Y)
AX[1].plot(x.flatten(), z.flatten(), '.', ms=1)
AX[0].axis('equal')
for ax, title in zip(AX, ['screen position', 'angular space']):
    pt.set_plot(ax, title=title)


# %% [markdown]
# ## Grating Stimuli

# %%
params = get_default_params('grating')

# a full-field grating (100deg => radius > screen-size)
for key, val in zip(['radius', 'x-center', 'y-center', 'angle'],
                    [100, 0, 0, 0]):
    params['%s' %key ] = val

fig, AX = pt.figure(axes=(3,1), ax_scale=(1.8,2), wspace=0, left=0, right=0)

for units, ax, title in zip(['deg', 'cm', 'lin-deg'], AX,
                           ['angular space', 'on screen', 'linearized angle display\n(deprecated)']):
    params['units'] = units
    stim = physion.visual_stim.stimuli.grating.stim(params)
    stim.plot_stim_picture(0, ax=ax, with_mask=True)
    ax.set_title(title)

# %%
params = get_default_params('grating')
params['no-window'], params['demo'] = True, False

# a full-field grating (100deg => radius > screen-size)
for key, val in zip(['radius', 'x-center', 'y-center', 'phase'],
                    [15, -30, 15, -90]):
    params['%s' %key ] = val

fig, AX = pt.figure(axes=(3,1), ax_scale=(1.8,2), wspace=0, left=0, right=0)

for units, ax, title in zip(['deg', 'cm', 'lin-deg'], AX,
                           ['angular space', 'on screen', 'linearized angle display\n(deprecated)']):
    params['units'] = units
    stim = physion.visual_stim.stimuli.grating.stim(params)
    stim.plot_stim_picture(0, ax=ax, with_mask=True)
    ax.set_title(title)

# %% [markdown]
# # 3-Screens U-shaped Configuration

# %% [markdown]
# 
# Screen 1:
# $$ x=-\frac{L}{2} $$
# Screen 2:
# $$ y=l_F $$
# Screen 3:
# $$ x=-\frac{L}{2} $$
# calculation [...]

# %%
import json
import physion
with open(os.path.join('..', 'src', 'physion', 'acquisition', 'protocols',
                       'demo', '3-screens.json')) as j:
    protocol = json.load(j)

stim = physion.visual_stim.build.build_stim(protocol)

fig, AX = pt.figure(axes=(1,2), ax_scale=(2,2))

for s in range(stim.screen['nScreens']):
    cond = stim.screen_ids.flatten()==(s+1)
    pt.scatter(stim.widths.flatten()[cond][::200],
                stim.heights.flatten()[cond][::200],
                ax=AX[0], ms=0.1, color=pt.tab10(s))
    pt.scatter(stim.x.flatten()[cond][::200],
                stim.z.flatten()[cond][::200],
                ax=AX[1], ms=0.2, color=pt.tab10(s))
    pt.annotate(AX[0], 'screen %i' % (s+1),
                (0.8-0.3*s, .99), ha='center',
                color=pt.tab10(s))

pt.set_plot(AX[0], xlabel='x (cm)', ylabel='y (cm)')
pt.set_plot(AX[1], xticks=[-90,0,90],
            ylabel='altitude (deg.)',
            xlabel='azimuth (deg.)')
for ax in AX:
    ax.axis('equal')
    ax.invert_xaxis()

# %% [markdown]
# ## 1) Properties

# %%
"""
Screen Dimension and Screen Placement Parameters for the Visual Stimulation
"""
screen_width = 48. # cm
screen_height = 27. # cm
distance_from_eye = 15. # cm

# %%
import sys
sys.path.append('../src')
from physion.utils import plot_tools as pt
import matplotlib.pylab as plt
import numpy as np

"""
Functions implementing basic trigonometric calculations
"""

def cm_to_angle(distance,
                distance_from_eye=15.):
    # distance_from_eye in cm
    return 180./np.pi*np.arctan(distance/distance_from_eye)

def cm_to_angle_lin(distance, distance_from_eye=15.):
    # the derivative of arctan in 0 is 1
    return distance/distance_from_eye*180./np.pi
    #return cm_to_angle(1, distance_from_eye=distance_from_eye)*distance

def angle_to_cm(angle,
                distance_from_eye=15.):
    # distance_from_eye in cm
    return distance_from_eye*np.tan(np.pi*angle/180.)

"""
plot
"""

max_height = cm_to_angle(screen_height/2.)
max_width = cm_to_angle(screen_width/2.)

angles = np.linspace(0, 1.3*max_width, 100) #
positions = angle_to_cm(angles, distance_from_eye=distance_from_eye)

fig, [ax, ax2, ax3] = plt.subplots(1, 3, figsize=(7,1.2))
plt.subplots_adjust(wspace=0.9)
ax.plot(angles, positions, color='k')
pt.plot(angles, angle_to_cm(1)*angles, ls='--', ax=ax, no_set=True, color='tab:red')
ax.annotate('lin.\napprox.', (0.99,0.35), color='tab:red', xycoords='axes fraction')

ax2.plot(positions, angles, color='k')
pt.plot(angle_to_cm(1)*angles, angles, ls='--', ax=ax2, no_set=True, color='tab:red')

ax.plot(np.ones(2)*max_height, [0,positions.max()], ':', color='tab:green', lw=1)
ax.annotate('max. \naltitude ', (max_height,positions.max()), color='tab:green', xycoords='data', va='top', ha='right')

ax.plot(np.ones(2)*max_width, [0,.98*positions.max()], ':', color='tab:blue', lw=1)
ax.annotate('max.\nazimuth', (max_width, positions.max()), color='tab:blue', xycoords='data', ha='center')

pt.set_plot(ax, 
            ylabel='distance (cm) \nfrom center',
            xlabel='angle (deg)\nfrom center',
            ylim=[0,positions.max()])
pt.set_plot(ax2, 
            xlabel='distance (cm) \nfrom center',
            ylabel='angle (deg)')

x = np.linspace(0, screen_width/2.)
ax3.plot(cm_to_angle_lin(x), cm_to_angle(x), 'k-')
ax3.plot(np.ones(2)*cm_to_angle_lin(x[-1]), [0,cm_to_angle(x[-1])], 'k:', lw=1)
ax3.plot([0,cm_to_angle_lin(x[-1])], np.ones(2)*cm_to_angle(x[-1]), 'k:', lw=1)
ax3.annotate('%.1f$^{o}$' % cm_to_angle_lin(x[-1]),
             (1, 0), xycoords='axes fraction')
ax3.annotate('%.1f$^{o}$' % cm_to_angle(x[-1]),
             (1, 1), xycoords='axes fraction', ha='right')

def angle_lin_to_true_angle(angle):
    return 180./np.pi*np.arctan(angle/180.*np.pi)

x = np.linspace(0, 90, 10)
ax3.plot(x, angle_lin_to_true_angle(x), 'ro')

pt.set_plot(ax3, 
            xlabel='lin. approx. angle (deg)',
            ylabel='true angle (deg)',
            num_xticks=5)

#ge.save_on_desktop(fig)

# %% [markdown]
# ### Plotting the screen with its angle scales:

# %%
import matplotlib.pylab as plt
fig, ax = plt.subplots(figsize=(screen_width/20.,screen_height/20.))
pt.draw_bar_scales(ax, Ybar=angle_to_cm(20), Xbar=angle_to_cm(20), Xbar_label='20$^o$', Ybar_label='20$^o$')
ax.set_xlim(np.arange(-1,3,2)*screen_width/2.)
ax.set_ylim(np.arange(-1,3,2)*screen_height/2.)
ax.set_xlabel('x-coord (cm)')
ax.set_ylabel('z-coord (cm)')
ax.set_title('screen coordinates')
#ge.save_on_desktop(fig)

# %% [markdown]
# ### Screen Dimensions in terms of Angles:

# %%
print('visual field covered:')
print('azimuth:', cm_to_angle(screen_width/2.))
print('altitude:', cm_to_angle(screen_height/2.))

# %% [markdown]
# ## 2) Plotting Visual Stimuli for Figures

# %%
# start from a simple protocol to load all required data
import json, os, sys
import numpy as np
import matplotlib.pylab as plt
with open(os.path.join('..', 'src', 'physion', 
                       'acquisition', 'protocols', 'demo', 
                       'moving-dots.json')) as j:
    protocol = json.load(j)
    
sys.path.append('../src')
from physion.visual_stim.build import build_stim
from physion.visual_stim.main import init_bg_image
protocol['no-window'] = True
stim = build_stim(protocol)

stim_index = 0
times = np.linspace(0.1, 3, 7)
fig, AX = plt.subplots(1, len(times), figsize=(8,1.2))

for i, t in enumerate(times):
    AX[i].imshow(-stim.get_image(stim_index, time_from_episode_start=t).T,
                 vmin=-1, vmax=1, cmap=plt.cm.binary)
    AX[i].set_title('t=%.1fs' % t, fontsize=7)
    AX[i].axis('off')

# %% [markdown]
# ## 3) Design of New Visual Stimuli

# %%
# start from a simple protocol to load all required data
import json, os
with open(os.path.join('..', 'src', 'physion', 'acquisition', 'protocols',
                       'demo', 'uniform-bg.json')) as j:
    protocol = json.load(j)
    
from physion.visual_stim.build import build_stim #, init_bg_image
from physion.visual_stim.main import init_bg_image
protocol['no-window'] = True
stim = build_stim(protocol)


# %%
def compute_new_image_with_dots(self, index, dot_size, Ndots,
                                bounds_range = [60,40]):

    within_bounds_cond = (np.abs(stim.x)<bounds_range[0]) & (np.abs(stim.z)<bounds_range[1])
    
    dots = []
    img = init_bg_image(self, index)
    for n in range(Ndots):
        iCenter = np.random.choice(np.arange(len(stim.x[within_bounds_cond])))
        x0 = stim.x[within_bounds_cond][iCenter]
        z0 = stim.z[within_bounds_cond][iCenter]
        stim.add_dot(img, (x0,z0), dot_size, -1)
    return img
        
img = compute_new_image_with_dots(stim, 0, 5, 8)
fig, ax = plt.subplots(1, figsize=(4,4))
ax.imshow(img.T, origin='lower', cmap=plt.cm.gray)
#ax.axis('off')

# %%
fig, ax = plt.subplots(1, figsize=(4,4))
radius = 20
img = init_bg_image(stim, 0)
img += np.exp(-(stim.x**2+stim.z**2)/2/radius**2)
ax.imshow(img.T, origin='lower', cmap=plt.cm.gray)
#ax.axis('off')

# %%
