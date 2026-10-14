import numpy as np

def set_angle_meshgrid_1screen(self, 
                               force_degree=False):
    """
    """

    # we start from the real pixel Cartesian coordinates on the screen
    widths, heights = np.meshgrid(\
                            np.linspace(-self.screen['width']/2., 
                                        self.screen['width']/2., 
                                        self.screen['resolution'][0]),
                            np.linspace(-self.screen['height']/2., 
                                        self.screen['height']/2., 
                                        self.screen['resolution'][1]),
                                    indexing='xy')
    # we transpose given our coordinate system:
    self.widths, self.heights = widths.T, heights.T

    self.screen_ids = np.ones(self.widths.shape, dtype=int)
    self.mask = np.ones(self.widths.shape, dtype=bool) # stim mask, True by default

    if force_degree or (self.units=='deg'):

        altitudeMax = np.arctan(self.screen['height']/2./self.screen['distance_from_eye'])
        azimuthMax = self.screen['resolution'][0]\
                            /self.screen['resolution'][1]*altitudeMax

        x, z = np.meshgrid(\
                        np.linspace(-azimuthMax, azimuthMax,
                                    self.screen['resolution'][0]),
                        np.linspace(-altitudeMax, altitudeMax,
                                    self.screen['resolution'][1]),
                                indexing='xy')
        self.x, self.z = x.T, z.T

        self.widths = self.screen['distance_from_eye']*np.tan(self.x)
        self.heights = self.screen['distance_from_eye']*np.tan(self.z)/np.cos(self.x)

        self.mask = (np.abs(self.widths)<=self.screen['width']/2.) &\
                        (np.abs(self.heights)<=self.screen['height']/2.)

    elif self.units=='cm':

        # we convert to angles in the x and z directions
        self.x = np.arctan(self.widths/self.screen['distance_from_eye'])
        self.z = np.arctan(self.heights*np.cos(self.x)/self.screen['distance_from_eye'])

    elif self.units=='lin-deg':

        # OLD STRATEGY --> deprecated >08/2024
        # we linearize the angle
        dAngle_per_pix = np.arctan(
                1./self.screen['resolution'][0]*self.screen['width']\
                /self.screen['distance_from_eye'])
        x, z = np.meshgrid(dAngle_per_pix*(\
                                np.arange(self.screen['resolution'][0])-\
                                    self.screen['resolution'][0]/2.),
                            dAngle_per_pix*(\
                                np.arange(self.screen['resolution'][1])-\
                                    self.screen['resolution'][1]/2.),
                                    indexing='xy')
        self.x, self.z = x.T, z.T


    # convert back to angles in degrees
    self.x *= 180./np.pi
    self.z *= 180./np.pi

def set_angle_meshgrid_U3Screens(self, 
                                 force_degree=False):
    """
    """

    X, Z = np.meshgrid(\
                np.linspace(-self.screen['width']/2., 
                            self.screen['width']/2., 
                            self.screen['resolution'][0]),
                    -self.screen['height_from_base']+\
                            np.linspace(0, self.screen['height'], 
                                    self.screen['resolution'][1]),
                        indexing='xy')

    L = self.screen['width']
    lF = self.screen['distance_front']

    # we transpose given our coordinate system:
    X, Z = X.T, Z.T

    widths = np.concatenate([X+(i-1)*L\
                                for i in range(self.screen['nScreens'])],
                                axis=0)
    heights = np.concatenate([Z for i in range(self.screen['nScreens'])],
                                axis=0)
    screen_ids = np.concatenate([np.array((i+1)+0*X, dtype=int)\
                                    for i in range(self.screen['nScreens'])],
                                axis=0)

    self.widths, self.heights = widths, heights
    self.screen_ids = screen_ids

    self.mask = np.ones(self.widths.shape, 
                        dtype=bool) # stim mask, True by default

    if force_degree or (self.units=='deg'):

        altitudeMax, altitudeMin = np.max(self.z), np.min(self.z)
        dZ = altitudeMax-altitudeMin
        azimuthMax = dZ*self.x.shape[0]/self.x.shape[1]/2.

        self.x, self.z = np.meshgrid(\
                     np.linspace(-azimuthMax, azimuthMax,
                                 self.x.shape[0]),
                     np.linspace(altitudeMin, altitudeMax,
                                  self.x.shape[1]),
                              indexing='ij')

    elif self.units=='cm':
        # by default, we go through the cm unit

        # we convert to angles in the x and z directions
        self.x, self.z = 0*self.widths, 0*self.widths

        # 
        #       screen by screen for the angular positions
        # 
        # - screen 1
        cond1 = (self.screen_ids==1)
        dX = self.widths[cond1]+L+(lF-L/2) # x-coordinates centered on -90 deg. angle
        self.x[cond1] = np.arctan(-L/2/dX)
        self.x[cond1] = (self.x[cond1]-np.pi)%np.pi-np.pi
        self.z[cond1] = np.arctan(\
            -2*self.heights[cond1]/L*np.sin(self.x[cond1]))

        # - screen 2
        cond2 = (self.screen_ids==2)
        self.x[cond2] = np.arctan(self.widths[cond2]/lF)
        self.z[cond2] = np.arctan(\
            self.heights[cond2]*np.cos(self.x[cond2])/lF)

        # - screen 3
        cond3 = (self.screen_ids==3)
        dX = self.widths[cond3]-L-(lF-L/2) # x-coordinates centered on 90 deg. angle
        aX = np.arctan(dX/(L/2.)) # alphaX angle
        self.x[cond3] = np.pi/2.+aX
        self.z[cond3] = np.arctan(np.sin(aX)*self.heights[cond3]/dX)


    # convert back to angles in degrees
    self.x *= 180./np.pi
    self.z *= 180./np.pi

def V2Screens_angles(X, Z, lF, screen_ids):
    """
    azimuth and altitude angles (radians) of the points (X, Z) of the V-shaped screens
        screen 1: y = X + lF (X<0),  screen 2: y = lF - X (X>0)
    see the "Visual-Stim-Design" notebook for the calculation
    """
    x, z = 0*X, 0*X

    # - screen 1
    cond1 = (screen_ids==1)
    # radius
    r = np.sqrt(
       X[cond1]**2 +\
        (X[cond1]+lF)**2 +\
            Z[cond1]**2 )
    # altitude angle
    z[cond1] = np.arcsin(Z[cond1]/r)
    # azimuth angle
    x[cond1] =\
        np.arctan(-X[cond1]/(X[cond1]+lF))
    # [0,2pi] -> [-pi,pi]
    cond1h = cond1 & (x<0)
    x[cond1h] += np.pi

    # - screen 2
    cond2 = (screen_ids==2)
    # radius
    r = np.sqrt(
       X[cond2]**2 +\
        (lF-X[cond2])**2 +\
            Z[cond2]**2 )
    # altitude angle
    z[cond2] = np.arcsin(Z[cond2]/r)
    # azimuth angle
    x[cond2] =\
        np.arctan(-X[cond2]/(lF-X[cond2]))
    # [0,2pi] -> [-pi,pi]
    cond2h = cond2 & (x>0)
    x[cond2h] -= np.pi

    return x, z


def set_angle_meshgrid_V2Screens(self,
                                 force_degree=False):
    """
    see the notebook for the calculation
    """

    L, H = self.screen['width'], self.screen['height']
    lF = self.screen['distance_front']
    hB = self.screen['height_from_base']

    xMax = lF+1./np.sqrt(2)*\
                    np.sqrt((L-lF/np.sqrt(2))**2)
    
    X, Z = np.meshgrid(\
                np.linspace(0, 1,
                            self.screen['resolution'][0]),
                np.linspace(0, 1,
                            self.screen['resolution'][1]),
                        indexing='xy')

    # we transpose given our coordinate system:
    X, Z = X.T, Z.T

    # X, Z = np.meshgrid(\
    #             np.linspace(-xMax, 0, 
    #                         self.screen['resolution'][0]),
    #                 -self.screen['height_from_base']+\
    #                         np.linspace(0, H,
    #                                 self.screen['resolution'][1]),
    #                     indexing='xy')

    self.X = np.concatenate([-xMax+i*xMax+xMax*X\
                                for i in range(self.screen['nScreens'])],
                                axis=0)
    self.Z = np.concatenate([-hB+H*Z\
                                for i in range(self.screen['nScreens'])],
                                axis=0)
    self.widths = np.concatenate([-L+i*L+L*X\
                                for i in range(self.screen['nScreens'])],
                                axis=0)
    self.heights = np.concatenate([-hB+H*Z\
                                for i in range(self.screen['nScreens'])],
                                axis=0)
    self.screen_ids = np.concatenate([np.array((i+1)+0*X, dtype=int)\
                                    for i in range(self.screen['nScreens'])],
                                axis=0)

    self.mask = np.ones(self.widths.shape,
                        dtype=bool) # stim mask, True by default

    if force_degree or (self.units=='deg'):

        # angles of the screen pixels (cm geometry), for the extent of the visual field
        _, z_screen = V2Screens_angles(self.X, self.Z, lF, self.screen_ids)

        # angular grid, linear in angle, with the same angle per pixel in both directions
        #   (as in the 1-screen 'deg' mode): the altitude spans that of the screens,
        #   the azimuth is centered on the front edge of the "V" (decreasing from
        #   screen 1 to screen 2, as in the 'cm' mode)
        altitudeMin, altitudeMax = np.min(z_screen), np.max(z_screen)
        dAngle = (altitudeMax-altitudeMin)/(self.X.shape[1]-1)
        azimuthMax = dAngle*(self.X.shape[0]-1)/2.

        self.x, self.z = np.meshgrid(\
                     np.linspace(azimuthMax, -azimuthMax,
                                 self.X.shape[0]),
                     np.linspace(altitudeMin, altitudeMax,
                                  self.X.shape[1]),
                              indexing='ij')

        # screen coordinates of the angular grid: intersection of the line of sight
        #   with the screen planes (screen 1: y=x+lF for x<0, screen 2: y=lF-x for x>0),
        #   azimuth convention of the 'cm' mode: tan(azimuth) = -x/y
        dx = -np.cos(self.z)*np.sin(self.x)
        dy = np.cos(self.z)*np.cos(self.x)
        dz = np.sin(self.z)
        on1 = (self.x>=0)          # positive azimuth -> screen 1
        with np.errstate(divide='ignore', invalid='ignore'):
            t = np.where(on1, lF/(dy-dx), lF/(dy+dx))   # distance along the line of sight
        Xs, Zs = t*dx, t*dz

        # screen-local coordinates, as in the 'cm' mode (widths: L*X/xMax, heights: Z)
        self.widths, self.heights = L*Xs/xMax, Zs
        self.mask = (t>0) &\
            np.where(on1, (Xs>=-xMax) & (Xs<=0), (Xs>=0) & (Xs<=xMax)) &\
            (Zs>=-hB) & (Zs<=H-hB)

    elif self.units=='cm':
        # by default, we go through the cm unit

        # we convert to angles in the x and z directions
        self.x, self.z = V2Screens_angles(self.X, self.Z, lF, self.screen_ids)

    # convert back to angles in degrees
    self.x *= 180./np.pi
    self.z *= 180./np.pi

def deg_per_pixel(screen, units='cm', image_shape=None,
                  where='center', screen_id=1):
    """
    visual angle (deg.) covered by one pixel, from the screen parameters (screens.py)
        and the pixel -> angle mapping of the protocol `units` (set_angle_meshgrid_* above)

    screen:      key of SCREENS (screens.py) or a screen dictionary
    units:       'cm' (default of the protocols), 'deg' or 'lin-deg'
    image_shape: (height, width) of an image stretched over the full screen
                    (as in the "natural_image" stimulus)
                    -> deg. per image pixel, otherwise deg. per screen pixel
    where:       'center': at the center of the screen (in 'cm' units, the pixel angle
                    decreases with eccentricity: largest value)
                 'mean': angular extent of the screen / number of pixels
    screen_id:   for multi-screen setups, the screen considered

    returns (along the width / azimuth, along the height / altitude)
    """
    from types import SimpleNamespace
    from physion.visual_stim.screens import SCREENS

    stim = SimpleNamespace(screen=SCREENS[screen] if isinstance(screen, str) else screen,
                           units=units)
    # same dispatch as visual_stim.set_angle_meshgrid (main.py)
    if stim.screen['nScreens']==1:
        set_angle_meshgrid_1screen(stim)
    elif stim.screen['nScreens']==2:
        set_angle_meshgrid_V2Screens(stim)
    else:
        set_angle_meshgrid_U3Screens(stim)

    # angles of the pixels of the considered screen: x varies along axis 0, z along axis 1
    resolution = stim.screen['resolution']
    cond = (stim.screen_ids==screen_id)
    x = stim.x[cond].reshape(resolution)
    z = stim.z[cond].reshape(resolution)

    if where=='center':
        i, j = resolution[0]//2, resolution[1]//2
        dx = np.abs(x[i+1, j]-x[i-1, j])/2.
        dz = np.abs(z[i, j+1]-z[i, j-1])/2.
    elif where=='mean':
        dx = (x[:, resolution[1]//2].max()-x[:, resolution[1]//2].min())/(resolution[0]-1)
        dz = (z[resolution[0]//2, :].max()-z[resolution[0]//2, :].min())/(resolution[1]-1)
    else:
        raise ValueError('where should be "center" or "mean"')

    if image_shape is not None:
        # image stretched over the screen: screen pixels per image pixel
        dx *= resolution[0]/image_shape[1]
        dz *= resolution[1]/image_shape[0]

    return dx, dz


if __name__=='__main__':

    import argparse, os, pathlib, json
    import physion.utils.plot_tools as pt
    import physion

    params = physion.visual_stim.build.get_default_params('grating')
    params['x-center'] = 35
    params['y-center'] = 10
    params['angle'] = 90
    params['radius'] = 225
    params['Screen'] = 'LN-VR-3screens'

    for units in ['deg', 'cm']:
        params['units'] = units
        stim = physion.visual_stim.build.build_stim(params)
        stim.plot_stim_picture(0, with_scale=True)

    pt.plt.show()

    # protocol['json_location'] = os.path.dirname(args.protocol)

    if False:
        # show the geometry 
        fig, AX = pt.figure(axes=(1,2), ax_scale=(2,2))

        for s in range(stim.screen['nScreens']):
            cond = stim.screen_ids.flatten()==(s+1)
            pt.scatter(stim.widths.flatten()[cond][::200],
                        stim.heights.flatten()[cond][::200],
                        ax=AX[0], ms=0.1, color=pt.tab10(s))
            
            pt.annotate(AX[0], 'screen #%i' % (s+1),
                        (stim.widths.flatten()[cond].min(),
                        stim.heights.flatten()[cond].max()),
                        xycoords='data', color = pt.tab10(s))

            pt.scatter(stim.x.flatten()[cond][::200],
                        stim.z.flatten()[cond][::200],
                        ax=AX[1], ms=0.2, color=pt.tab10(s))

        pt.set_plot(AX[0], xlabel='x (cm)', ylabel='y (cm)')
        pt.set_plot(AX[1], xticks=[-90,0,90],
                    ylabel='altitude (deg.)',
                    xlabel='azimuth (deg.)')
        for ax in AX:
            ax.axis('equal')

        pt.plt.show()
