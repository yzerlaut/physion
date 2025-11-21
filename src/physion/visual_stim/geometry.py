import numpy as np

def set_angle_meshgrid_1screen(self):
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

    if self.units=='cm':

        # we convert to angles in the x and z directions
        self.x = np.arctan(self.widths/self.screen['distance_from_eye'])
        self.z = np.arctan(self.heights*np.cos(self.x)/self.screen['distance_from_eye'])

    elif self.units=='deg':

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

def set_angle_meshgrid_U3Screens(self):
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

    # if self.units=='cm':
    if True:
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

    if self.units=='deg':

        altitudeMax, altitudeMin = np.max(self.z), np.min(self.z)
        dZ = altitudeMax-altitudeMin
        azimuthMax = dZ*self.x.shape[0]/self.x.shape[1]/2.

        self.x, self.z = np.meshgrid(\
                     np.linspace(-azimuthMax, azimuthMax,
                                 self.x.shape[0]),
                     np.linspace(altitudeMin, altitudeMax,
                                  self.x.shape[1]),
                              indexing='ij')

    # convert back to angles in degrees
    self.x *= 180./np.pi
    self.z *= 180./np.pi

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
