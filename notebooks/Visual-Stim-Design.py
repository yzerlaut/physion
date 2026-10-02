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

# %%
from IPython.display import SVG, display
display(SVG("../docs/visual_stim/coordinates.svg"))

# %% [markdown]
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
# where $\theta$ is the zenith angle, $r$ the radius and $\phi$ the azimuth angle.
#
# The relationship with our coordinates are: $\theta_z = \pi/2 - \theta$ (the altitude angle) and $\theta_x = \phi - \pi / 2$ (the azimuth angle).
#
#
# Using [trigonometric identities](https://en.wikipedia.org/wiki/List_of_trigonometric_identities), we get:
#
# $$ x = r \, \cos(\theta_z) \, \sin(\theta_x) $$
#
# $$ y = - r \, \cos(\theta_z) \, \cos(\theta_x) $$
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
# r = - \frac{C}{ \cos(\theta_z) \, \cos(\theta_x) }
# $$
#
# So:
#
# $$ x = - \frac{C}{ \cos(\theta_z) \, \cos(\theta_x)} \, \cos(\theta_z) \, \sin(\theta_x)  = - C \, \tan(\theta_x) $$
#
# $$ z = - \frac{C}{ \cos(\theta_z) \, \cos(\theta_x)} \, \sin(\theta_z)  = - C \, \frac{\tan(\theta_z)}{\cos(\theta_x)} $$
#
# ------------------------
#
# This is implemented in [visual_stim/geometry.py](../../visual_stim/geometry.py) in the function `set_angle_meshgrid_1screen`

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
# # 2-Screens V-shaped Configuration

# %% [markdown]
# 
# ### For both screens
#
# $$ z \in [-h_B, L-h_B] $$
# ##### To find $\theta_z$, we use the expression:
# $$ \theta_z = \arcsin( \frac{z}{r} ) $$
# with: $$ r = \sqrt{ x^2 + y^2 + z^2 } $$
#
# 
# ### Screen 1: (x<0)
# $$ x \in [- l_F - \frac{1}{\sqrt{2}} \cdot \sqrt{ (L-\frac{ l_F }{ \sqrt{2} })^2 }, 0] $$
# with :
# $$ y=x + d_f  $$
# ##### To find $\theta_x$, we use the expression:
# $$ \theta_x = \arctan(-\frac{x}{x+l_F}) $$
#
# ### Screen 2: (x>0)
# $$ x \in [0, l_F + \frac{1}{\sqrt{2}} \cdot \sqrt{ (L-\frac{ l_F }{ \sqrt{2} })^2 }] $$
# with:
# $$ y= d_f -x $$
# ##### To find $\theta_x$, we use the expression:
# $$ \theta_x = \arctan(-\frac{x}{l_F-x}) $$
#
# This is implemented in [visual_stim/geometry.py](../../visual_stim/geometry.py) in the function `set_angle_meshgrid_V2screens`

# %%
import json, sys, os
sys.path += ['../src']
from physion.visual_stim.build import build_stim
import physion.utils.plot_tools as pt
with open(os.path.join('..', 'src', 'physion', 'acquisition', 'protocols',
                       'demo', '2-screens.json')) as j:
    protocol = json.load(j)

stim = build_stim(protocol)

fig, AX = pt.figure(axes=(1,2), ax_scale=(2,2))

for s in range(stim.screen['nScreens']):
    cond = stim.screen_ids.flatten()==(s+1)
    # screen pixels
    pt.scatter(stim.widths.flatten()[cond][::200],
                stim.heights.flatten()[cond][::200],
                ax=AX[0], ms=0.1, color=pt.tab10(s))
    # screen angles
    pt.scatter(stim.x.flatten()[cond][::200],
                stim.z.flatten()[cond][::200],
                ax=AX[1], ms=0.2, color=pt.tab10(s))
    pt.annotate(AX[0], 'screen %i' % (s+1),
                (0.3+0.3*s, .99), ha='center',
                color=pt.tab10(s))

pt.set_plot(AX[0], xlabel='x (cm)', ylabel='z (cm)')
pt.set_plot(AX[1], xticks=[-90, -45, 0, 45, 90],
            ylabel='altitude (deg.)',
            xlabel='azimuth (deg.)')
for ax in AX:
    ax.axis('equal')
AX[1].invert_xaxis()


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
