"""
class for the visual stimulation

color in grayscale in range [0,1]
"""

import numpy as np
import itertools
import os
import pathlib
import time
import json

from physion.visual_stim.screens import SCREENS
from physion.visual_stim.build import build_stim
from physion.visual_stim.shuffling import *

defaults = {\
    'presentation-duration': 2,
    'presentation-prestim-period': 6,
    'presentation-poststim-period': 6,
    'presentation-interstim-period': 4,
    'presentation-blank-screen-color':0.5,
    'movie_refresh_freq':30.,
    # 'units':'lin-deg',
    'units':'cm',
}

class visual_stim:
    """
    """

    def __init__(self,
                 protocol,
                 default_params={},
                 demo=False):
        """
        """

        self.protocol = protocol

        # initialize screen parameters
        self.screen = SCREENS[self.protocol['Screen']]
        self.k = self.screen['gamma_correction']['k']
        self.gamma = self.screen['gamma_correction']['gamma']

        # insure backward compatibility, by setting params if not existing
        for key in defaults:
            if key not in self.protocol:
                self.protocol[key] = defaults[key]

        self.units = self.protocol['units']
        self.movie_refresh_freq = self.protocol['movie_refresh_freq']
        self.blank_color=self.gamma_correction(\
                self.protocol['presentation-blank-screen-color'])

        if demo or (('demo' in self.protocol) and self.protocol['demo']):
            self.screen['fullscreen'] = False

        # then we can initialize the angle
        self.set_angle_meshgrid()

        ### INITIALIZE EXP ###
        if not (self.protocol['Presentation']=='multiprotocol'):
            self.init_experiment(protocol, default_params)


    ################################
    #  ---       input/output  --- #
    ################################
    def save(self, folder):
        with open(os.path.join(folder, 'protocol.json'),
                               'w', encoding='utf-8') as f:
            json.dump(self.protocol, f,
                      ensure_ascii=False, indent=4)
        print('[ok] visual-stimulation protocol saved as "%s"' %\
                os.path.join(folder, 'protocol.json'))
        np.save(os.path.join(folder, 'visual-stim.npy'), self.experiment)
        print('[ok] visual-stimulation time course saved as "%s"' %\
                os.path.join(folder, 'visual-stim.npy'))

    ################################
    #  ---   Gamma correction  --- #
    ################################

    def gamma_correction(self, lum):
        return np.power(lum/self.k, 1./self.gamma)

    ################################
    #  ---       Geometry      --- #
    ################################

    def set_angle_meshgrid(self):
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


    # some general grating functions
    def compute_rotated_coords(self, angle,
                               xcenter=0, zcenter=0):
        return (self.x-xcenter)*np.cos(angle/180.*np.pi)+\
                    (self.z-zcenter)*np.sin(angle/180.*np.pi)


    def compute_grating(self, xrot,
                        spatial_freq=0.1, 
                        time_phase=0.,
                        phase_shift_Deg=90.):
        return 0.5+np.cos(phase_shift_Deg*np.pi/180.+\
                        2*np.pi*(spatial_freq*xrot-time_phase))/2.


    ################################
    #  ---  Draw Stimuli       --- #
    ################################

    def add_grating_patch(self, image,
                          angle=0,
                          radius=10,
                          spatial_freq=0.1,
                          contrast=1.,
                          phase_shift_Deg=0.,
                          time_phase=0.,
                          xcenter=0,
                          zcenter=0):
        """ add a grating patch, drifting when varying the time phase"""
        xrot = self.compute_rotated_coords(angle,
                                           xcenter=xcenter,
                                           zcenter=zcenter)

        cond = ((self.x-xcenter)**2+(self.z-zcenter)**2)<radius**2

        full_grating = self.compute_grating(xrot,
                                            spatial_freq=spatial_freq,
                                            time_phase=time_phase,
                                            phase_shift_Deg=phase_shift_Deg)-0.5

        image[cond] += contrast*full_grating[cond] 

    def add_gaussian(self, image,
                     t=0, t0=0, sT=1.,
                     radius=10,
                     amplitude=0.5,
                     xcenter=0,
                     zcenter=0):
        """
        add a gaussian luminosity increase
        """
        image += np.exp(-((self.x-xcenter)**2+(self.z-zcenter)**2)/2./radius**2)*\
                     amplitude*np.exp(-(t-t0)**2/2./sT**2)


    def add_dot(self, image, pos, size, color, type='square'):
        """
        add dot, either square or circle
        """
        if type=='square':
            cond = (self.x>(pos[0]-size/2)) & (self.x<(pos[0]+size/2)) &\
                    (self.z>(pos[1]-size/2)) & (self.z<(pos[1]+size/2))
        else:
            cond = np.sqrt((self.x-pos[0])**2+(self.z-pos[1])**2)<size
        image[cond] = color

    def blank_surround(self, image,
                      radius=10,
                        xcenter=0,
                          zcenter=0,
                            bg_color=0.5):
        """ blank surround """

        cond = ((self.x-xcenter)**2+(self.z-zcenter)**2)<radius**2

        image[~cond] = bg_color

        return image

    ########################################################
    #  ---     Experiment (time course) properties     --- #
    ########################################################

    def init_experiment(self, 
                        protocol, 
                        default_params):

        self.experiment = {}

        if protocol['Presentation']=='Single-Stimulus':

            # ------------    SINGLE STIMS ------------
            # 
            self.experiment['index'] = [0]
            self.experiment['repeat'] = [0]
            self.experiment['bg-color'] = \
                    [protocol['presentation-blank-screen-color']]
            self.experiment['time_start'] = [\
                    protocol['presentation-prestim-period']]
            self.experiment['time_stop'] = [\
                    protocol['presentation-duration']+\
                    protocol['presentation-prestim-period']]
            self.experiment['time_duration'] = [\
                    protocol['presentation-duration']]
            self.experiment['interstim'] = [\
                protocol['presentation-interstim-period']\
                if 'presentation-interstim-period' in protocol else 0]
            # 
            for key in default_params:
                if key in protocol:
                    self.experiment[key] = [protocol[key]]
                else:
                    self.experiment[key] = [default_params[key]]

        else:

            #
            # ------------  MULTIPLE STIMS ------------
            # 

            #############################################
            ###    == build all set of parameters ==  ### 
            #############################################

            VECS, FULL_VECS = [], {}
            for key in default_params:
                FULL_VECS[key], self.experiment[key] = [], []


                if ('N-log-'+key in protocol) and (protocol['N-log-'+key]>1):
                    # LOG-SPACED parameters
                    VECS.append(np.logspace(np.log10(protocol[key+'-1']),
                                            np.log10(protocol[key+'-2']),
                                            protocol['N-log-'+key]))
                elif ('N-'+key in protocol) and (protocol['N-'+key]>1):
                    # LIN-SPACED parameters
                    VECS.append(np.linspace(protocol[key+'-1'],
                                            protocol[key+'-2'],
                                            protocol['N-'+key]))
                elif key in protocol:
                    # parameter in the protocol:
                    VECS.append(np.array([protocol[key]])) 
                else:
                    # just the default value:
                    VECS.append(np.ones(1)*default_params[key])


            for vec in itertools.product(*VECS):
                for i, key in enumerate(default_params.keys()):
                    FULL_VECS[key].append(vec[i])


            #############################################
            ###    == build repetition sequence   ==  ### 
            #############################################

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # then dealing with repetitions
            Nrepeats = max([1, protocol['N-repeat']])

            # episodes in order of stim variations
            indices = np.concatenate(
                    [index_no_repeat for n in range(Nrepeats)])
            repeats = np.concatenate(
                    [np.ones(len(index_no_repeat),dtype=int)*n\
                            for n in range(Nrepeats)])

            #############################################
            ###    ======     SHUFFLING ?      =====  ### 
            #############################################

            if ('shuffling' in protocol) or\
                    (protocol['Presentation']=='Randomized-Sequence'):

                indices, repeats = shuffle_single_protocol(indices, 
                                                           repeats,
                                                           protocol)

            #############################################
            ###    ==  building the time course   ==  ### 
            #############################################

            for k in ['index', 'repeat', 'bg-color', 'interstim',
                      'time_start', 'time_stop', 'time_duration']:
                self.experiment[k] = []

            for n, i, r in zip(range(len(indices)), indices, repeats):

                for key in default_params:
                    self.experiment[key].append(FULL_VECS[key][i])

                self.experiment['index'].append(i) # shuffled
                self.experiment['bg-color'].append(self.blank_color)
                self.experiment['repeat'].append(repeats[i])

                if n==0:
                    self.experiment['time_start'].append(\
                            protocol['presentation-prestim-period'])
                else:
                    self.experiment['time_start'].append(\
                        self.experiment['time_stop'][-1]+\
                                protocol['presentation-interstim-period'])

                self.experiment['time_stop'].append(\
                    self.experiment['time_start'][-1]+\
                            protocol['presentation-duration'])

                self.experiment['interstim'].append(\
                        protocol['presentation-interstim-period'])
                self.experiment['time_duration'].append(\
                        protocol['presentation-duration'])

        for k in ['index', 'repeat','time_start', 'time_stop',
                  'bg-color', 'interstim', 'time_duration']:
            self.experiment[k] = np.array(self.experiment[k])

        # we add a protocol_id
        # 0 by default for single protocols, overwritten for multiprotocols
        self.experiment['protocol_id'] = np.zeros(\
                len(self.experiment['index']), dtype=int)
        # we write a tstop 
        self.tstop = self.experiment['time_stop'][-1]+\
                            protocol['presentation-poststim-period']


    def prepare_stimProps_tables(self,
                                 verbose=True):
            if verbose:
                tic = time.time()

            dt = 1./self.movie_refresh_freq
            # build time axis
            #               add 2 seconds at the end for the end-stim flag
            t = np.arange(int((2+self.tstop)/dt))*dt 
            # array for the interstim flag
            self.is_interstim = np.ones(len(t), dtype=bool) # default to True
            self.next_index_table = np.zeros(len(t), dtype=int)
            # array for the time start
            self.time_start_table = np.zeros(len(t), dtype=float)
            # -- loop over episode
            for i in range(len(self.experiment['index'])):
                tCond = (t>=self.experiment['time_start'][i]) &\
                            (t<self.experiment['time_stop'][i])
                self.is_interstim[tCond] = False
                self.time_start_table[tCond] = self.experiment['time_start'][i]
                self.next_index_table[t>=self.experiment['time_start'][i]] = i+1
            # flag for end of stimulus
            self.next_index_table[t>=self.experiment['time_stop'][-1]] = -1
            if verbose:
                print('[ok] stim tables initialisation took: %.2f' % (\
                        time.time()-tic)) 



    #################################################
    #############    DRAWING STIMULI   ##############
    #################################################

    def get_image(self, episode, time_from_episode_start=0):
        """
        print('to be implemented in child class')
        """
        return 0*self.x+0.5

    def get_prestim_image(self):
        if 'presentation-prestim-screen' in self.protocol:
            return (1+self.protocol['presentation-prestim-screen'])/2.+\
                    0*self.x
        else:
            return 0*self.x

    def get_interstim_image(self):
        if 'presentation-interstim-screen' in self.protocol:
            return (1+self.protocol['presentation-interstim-screen'])/2.+\
                    0*self.x
        else:
            return 0*self.x

    def get_poststim_image(self):
        if 'presentation-poststim-screen' in self.protocol:
            return (1+self.protocol['presentation-poststim-screen'])/2.+\
                    0*self.x
        else:
            return 0*self.x

    def image_to_frame(self, img, norm=False, psychopy_to_numpy=False):
        """ need to transpose given the current coordinate system"""
        if psychopy_to_numpy:
            return img.T/2.+0.5
        if norm:
            return (img.T-np.min(img))/(np.max(img)-np.min(img)+1e-6)
        else:
            return img.T

    def get_vse(self, episode):
        """
        Virtual Scene Exploration dictionary
        None by default, should be overriden by method in children class
        """
        return None

    def show_frame(self, episode,
                   time_from_episode_start=0,
                   label={'size':10, 'label':'10$^o$ ',
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
                   arrow=None,
                   vse=False,
                   with_mask=False,
                   ax=None,
                   return_img=False):
        """

        display the visual stimulus at a given time in a given episode of a stimulation pattern

        --> optional with angular label (switch to None to remove)
                   label={'size':5,label='5deg',
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
        --> optional with arrow for direction propagation (switch to None to remove)
                   arrow={'direction':90,
                          'center':(0,0),
                          'length':10,
                          'width_factor':0.05,
                          'color':'red'},
        --> optional with virtual scene exploration trajectory (switch to None to remove)
        """

        if ax==None:
            import matplotlib.pylab as plt
            fig, ax = plt.subplots(1,
                       figsize=(4,
                                4*self.screen['resolution'][1]/self.screen['resolution'][0]))

        image = self.get_image(episode,
                                  time_from_episode_start=\
                                           time_from_episode_start)

        if with_mask:
            image[~self.mask] = 1

        img = ax.imshow(self.image_to_frame(image,
                                            psychopy_to_numpy=True),
                        extent=(0, self.screen['resolution'][0],
                                0, self.screen['resolution'][1]),
                        cmap='gray',
                        vmin=0, vmax=1,
                        origin='lower',
                        aspect='equal')


        if vse:
            self.vse = self.get_vse(episode)
            if self.vse is not None:
                self.add_vse(ax, self.vse)

        ax.axis('off')
        ax.axis('equal')

        if label is not None:
            nz, nx = self.x.shape
            L, shift = nx/(self.z.max()-self.z.min())*label['size'], label['shift_factor']*nx
            ax.plot([-shift, -shift], [-shift,L-shift], 'k-', lw=label['lw'])
            ax.plot([-shift, L-shift], [-shift,-shift], 'k-', lw=label['lw'])
            ax.annotate(label['label'], (-shift, -shift), 
                        fontsize=label['fontsize'], ha='right', va='bottom')

        if return_img:
            return img
        else:
            return ax

    def plot_stim_picture(self, episode, 
                          ax=None,
                          label=None,
                          vse=False,
                          with_scale=False,
                          arrow={'length':20,
                                 'width_factor':0.05,
                                 'color':'red'},
                          with_mask=False):

        """
        """
        tcenter = .5*(self.experiment['time_stop'][episode]-\
                      self.experiment['time_start'][episode])

        
        if with_scale:
            if self.units in ['cm', 'lin-deg']:
                label={'size':10/self.heights.max()*self.z.max(),
                       'label':'10cm ',
                       'shift_factor':0.02,
                       'lw':1, 'fontsize':10}
            else:
                label={'size':20,'label':'20$^o$  ',
                       'shift_factor':0.02,
                       'lw':1, 'fontsize':10}
        else:
            label=None

        ax = self.show_frame(episode, tcenter, 
                             ax=ax,
                             label=label,
                             with_mask=with_mask)
        return ax


    def update_frame(self, episode, img,
                     time_from_episode_start=0):
        img.set_array(self.image_to_frame(self.get_image(episode,
                                                time_from_episode_start=\
                                                    time_from_episode_start),
                                          psychopy_to_numpy=True))


    def add_arrow(self, arrow, ax):
        nx, nz = self.x.shape
        ax.arrow(self.angle_to_pix(arrow['center'][0])+nx/2,
                 self.angle_to_pix(arrow['center'][1])+nz/2,
                 np.cos(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 np.sin(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 width=self.angle_to_pix(arrow['length'])*arrow['width_factor'],
                 color=arrow['color'])

    def add_vse(self, ax, vse):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot([self.screen['resolution'][0]/2.]+list(vse['x']),
                [self.screen['resolution'][1]/2.]+list(vse['y']), 'o-', color='#d62728', lw=0.5, ms=2)


#####################################################
##  ----         MULTI-PROTOCOLS            --- #####
#####################################################

class multiprotocol(visual_stim):

    def __init__(self, protocol):

        super().__init__(protocol,
                         demo=(('demo' in protocol) and protocol['demo']))

        self.STIM, i = [], 1

        if 'Protocol-1-Stimulus' in protocol:
            print('     - loading visual stim params from saved protocol params [...]')
            # this means the subprotocol parameters were already saved, 
            #      so we build the protocol from those
            while 'Protocol-%i'%i in protocol:
                subprotocol = {'Screen':protocol['Screen'],
                               'Presentation':''}
                for key in protocol:
                    if ('Protocol-%i-'%i in key):
                        nKey = key.replace('Protocol-%i-'%i, '')
                        subprotocol[nKey] = protocol[key]
                self.STIM.append(build_stim(subprotocol))
                i+=1
        else:
            # we generate a new multiprotocol
            while 'Protocol-%i'%i in protocol:
                path_list = [pathlib.Path(__file__).resolve().parents[1],
                            'acquisition', 
                             'protocols']+protocol['Protocol-%i'%i].split('/')
                Ppath = os.path.join(*path_list)
                if not os.path.isfile(Ppath):
                    print(' [!!] "%s" not found in Protocol folder [!!]   ' %\
                                            protocol['Protocol-%i'%i])
                with open(Ppath, 'r') as fp:
                    subprotocol = json.load(fp)
                    # properties inherited form the parent protocol:
                    for key in ['Screen', 'units', 'movie_refresh_freq']:
                        subprotocol[key] = protocol[key]
                    self.STIM.append(build_stim(subprotocol))
                    for key, val in subprotocol.items():
                        protocol['Protocol-%i-%s'%(i,key)] = val
                i+=1

        self.experiment = {'protocol_id':[]}

        # we initialize the keys of all stims
        for stim in self.STIM:
            for key in stim.experiment:
                self.experiment[key] = []

        # then we iterate over values
        for IS, stim in enumerate(self.STIM):
            for i in range(len(stim.experiment['index'])):
                for key in self.experiment:
                    if (key in stim.experiment) and (key!='protocol_id'):
                        self.experiment[key].append(stim.experiment[key][i])
                    elif key in ['interstim-screen']:
                        # if not in keys, mean 0 interstim (e.g. sparse noise stim.)
                        self.experiment[key].append(0) 
                    elif key not in ['protocol_id', 'time_duration']:
                        self.experiment[key].append(None)
                self.experiment['protocol_id'].append(IS)

        # ---------------------------- #
        # # SHUFFLING IF NECESSARY
        # ---------------------------- #

        indices = shuffle_multiprotocol(\
             np.arange(len(self.experiment['index'])),
                    self.experiment['repeat'], 
                              self.experiment['protocol_id'],
                                    protocol) # nothing if no shuffling key

        # updating sequence
        for key in self.experiment:
            if key not in ['time_start', 'time_stop', 
                           'interstim', 'time_duration']:
                print(key)
                self.experiment[key] = \
                    np.array(self.experiment[key])[indices]

        # rebuilding experiment time course
        for n, i, isi, dur in zip(range(len(indices)), indices, 
                               np.array(self.experiment['interstim'])[indices],
                               np.array(self.experiment['time_duration'])[indices]):

            if n==0:
                self.experiment['time_start'].append(\
                        protocol['presentation-prestim-period'])
            else:
                self.experiment['time_start'].append(\
                    self.experiment['time_stop'][-1]+isi)

            self.experiment['time_stop'].append(\
                                self.experiment['time_start'][-1]+dur)
            self.experiment['interstim'].append(isi)

    ##############################################
    ##  ----  MAPPING TO CHILD PROTOCOLS --- #####
    ##############################################

    # functions implemented in child class
    def get_image(self, index, time_from_episode_start=0):
        return getattr(self.STIM[self.experiment['protocol_id'][index]],
                       'get_image')(self.experiment['index'][index],
                                    time_from_episode_start=time_from_episode_start)

    def get_frames_sequence(self, index):
        return getattr(self.STIM[self.experiment['protocol_id'][index]],
                       'get_frames_sequence')(self.experiment['index'][index])

    def plot_stim_picture(self, index, 
                          ax=None, label=None, vse=False):
        return getattr(self.STIM[self.experiment['protocol_id'][index]],
                       'plot_stim_picture')(self.experiment['index'][index],
                                            ax=ax, label=label, vse=vse)

    def get_vse(self, index, ax=None, label=None, vse=False):
        return getattr(self.STIM[self.experiment['protocol_id'][index]],
                       'get_vse')(self.experiment['index'][index])


#####################################
##  ----  BUILDING STIMULI  --- #####
#####################################

def init_bg_image(cls, index):
    """ initializing an empty image"""
    return cls.experiment['bg-color'][index]+0.*cls.x


if __name__=='__main__':

    print(5)
    stim = visual_stim()
    print(4)
