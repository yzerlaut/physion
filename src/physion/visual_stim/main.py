"""
class for the visual stimulation

- test with :
python -m physion.visual_stim.main physion/acquisition/protocols/drifting-gratings.json

N.B. Psychopy has colors between -1 (black) and +1 (white)
"""
import numpy as np
import itertools
import os
import cv2
import pathlib
import time
import json

from physion.visual_stim.screens import SCREENS
from physion.visual_stim.build import build_stim


class visual_stim:
    """
    """

    def __init__(self,
                 protocol,
                 keys=[], # need to pass the varied parameters
                 units='deg', # degree vs cm, cm -> to show on the screen !
                 demo=False):
        """
        """
        self.protocol = protocol

        # initialize screen parameters
        self.screen = SCREENS[self.protocol['Screen']]
        self.k = self.screen['gamma_correction']['k']
        self.gamma = self.screen['gamma_correction']['gamma']
        self.blank_color=self.gamma_corrected_lum(\
                2*self.protocol['presentation-blank-screen-color']-1)
        self.units = units

        if demo or (('demo' in self.protocol) and self.protocol['demo']):
            # --------------------- #
            #  ---- DEMO MODE ---- ##    we override the parameters
            # --------------------- #
            sr0, sr1 = self.screen['resolution']
            self.screen['resolution'] = (800, int(800*sr1/sr0))
            self.screen['screen_id'] = 0
            self.screen['fullscreen'] = False

        # then we can initialize the angle
        self.set_angle_meshgrid()
        # and the screen presentation if need
        if not ('no-window' in self.protocol):
            self.init_screen_presentation()

        ### INITIALIZE EXP ###
        if not (self.protocol['Presentation']=='multiprotocol'):
            self.init_experiment(protocol, keys)


    ################################################
    ###  Initialize the PsychoPy window         ####
    ################################################

    def init_screen_presentation(self):

        self.win = visual.Window(self.screen['resolution'],
                                 fullscr=self.screen['fullscreen'],
                                 units='pix',
                                 screen=1 if (('Rig' in self.protocol) and\
                                         ('A1' in self.protocol['Rig'])) else 2,
                                 checkTiming=(os.name=='posix'), # for os x
                                 color=self.blank_color)


    ################################
    #  ---   Gamma correction  --- #
    ################################

    def gamma_corrected_lum(self, level):
        return 2*np.power(((level+1.)/2./self.k), 1./self.gamma)-1.

    def gamma_corrected_contrast(self, contrast):
        return np.power(contrast/self.k, 1./self.gamma)

    ################################
    #  ---       Geometry      --- #
    ################################

    def cm_to_angle(self, value):
        return 180./np.pi*np.arctan(value/self.screen['distance_from_eye'])

    def pix_to_angle(self, value):
        return self.cm_to_angle(value/self.screen['resolution'][0]*self.screen['width'])

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
        return (self.x-xcenter)*np.cos(angle/180.*np.pi)+(self.z-zcenter)*np.sin(angle/180.*np.pi)

    def compute_grating(self, xrot,
                        spatial_freq=0.1, 
                        time_phase=0.,
                        phase_shift_Deg=90.):
        return (1+np.cos(phase_shift_Deg*np.pi/180.+\
                            2*np.pi*(spatial_freq*xrot-time_phase)))/2.

    ################################
    #  ---  Draw Stimuli       --- #
    ################################

    def add_grating_patch(self, image,
                          angle=0,
                          radius=10,
                          spatial_freq=0.1,
                          contrast=1.,
                          phase=0.,
                          xcenter=0,
                          zcenter=0):
        """ add a grating patch, drifting when varying the time phase"""
        xrot = self.compute_rotated_coords(angle,
                                           xcenter=xcenter,
                                           zcenter=zcenter)

        cond = ((self.x-xcenter)**2+(self.z-zcenter)**2)<radius**2

        print(phase)
        full_grating = self.compute_grating(xrot,
                                            spatial_freq=spatial_freq,
                                            phase_shift_Deg=phase)-0.5

        image[cond] = 2*contrast*full_grating[cond] # /!\ "=" for the patch



    def add_gaussian(self, image,
                     t=0, t0=0, sT=1.,
                     radius=10,
                     contrast=1.,
                     xcenter=0,
                     zcenter=0):
        """
        add a gaussian luminosity increase
        N.B. when contrast=1, you need black background, otherwise it will saturate
             when contrast=0.5, you can start from the grey background to reach white in the center
        """
        image += 2*np.exp(-((self.x-xcenter)**2+(self.z-zcenter)**2)/2./radius**2)*\
                     contrast*np.exp(-(t-t0)**2/2./sT**2)


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


    ########################################################
    #  ---     Experiment (time course) properties     --- #
    ########################################################

    def init_experiment(self, protocol, keys):

        self.experiment = {}

        if protocol['Presentation']=='Single-Stimulus':

            # ------------    SINGLE STIMS ------------
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
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

        else:

            # ------------  MULTIPLE STIMS ------------
            VECS, FULL_VECS = [], {}
            for key in keys:
                FULL_VECS[key], self.experiment[key] = [], []
                if ('N-log-'+key in protocol) and (protocol['N-log-'+key]>1):
                    # LOG-SPACED parameters
                    VECS.append(np.logspace(np.log10(protocol[key+'-1']),
                                            np.log10(protocol[key+'-2']),
                                            protocol['N-log-'+key]))
                elif protocol['N-'+key]>1:
                    # LIN-SPACED parameters
                    VECS.append(np.linspace(protocol[key+'-1'],
                                            protocol[key+'-2'],
                                            protocol['N-'+key]))
                else:
                    #  /!\ we pick the SECOND VALUE as the constant one 
                    #         (so remember to fill this right in GUI)
                    VECS.append(np.array([protocol[key+'-2']])) 

            for vec in itertools.product(*VECS):
                for i, key in enumerate(keys):
                    FULL_VECS[key].append(vec[i])

            for k in ['index', 'repeat', 'time_start', 'time_stop',
                      'bg-color', 'interstim', 'time_duration']:
                self.experiment[k] = []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # then dealing with repetitions
            Nrepeats = max([1, protocol['N-repeat']])

            if 'shuffling-seed' in protocol:
                np.random.seed(protocol['shuffling-seed']) # initialize random seed

            for r in range(Nrepeats):

                # shuffling if necessary !
                if (protocol['Presentation']=='Randomized-Sequence'):
                    np.random.shuffle(index_no_repeat)

                for n, i in enumerate(index_no_repeat):
                    for key in keys:
                        self.experiment[key].append(FULL_VECS[key][i])
                    self.experiment['index'].append(i) # shuffled
                    # self.experiment['bg-color'].append(self.blank_color)
                    self.experiment['repeat'].append(r)
                    self.experiment['time_start'].append(\
                            protocol['presentation-prestim-period']+\
                            (r*len(index_no_repeat)+n)*\
                                (protocol['presentation-duration']+\
                                protocol['presentation-interstim-period']))
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

        if len(self.experiment['bg-color'])!=len(self.experiment['index']):
            self.experiment['bg-color'] = self.blank_color*\
                    np.ones(len(self.experiment['index']))

        # we add a protocol_id
        # 0 by default for single protocols, overwritten for multiprotocols
        self.experiment['protocol_id'] = np.zeros(\
                len(self.experiment['index']), dtype=int)
        # we write a tstop 
        self.tstop = self.experiment['time_stop'][-1]+\
                            protocol['presentation-poststim-period']


    def prepare_stimProps_tables(self, dt, 
                                 verbose=True):
            if verbose:
                tic = time.time()
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
                print(' [ok] stim tables initialisation took: %.2f' % (\
                        time.time()-tic)) 

    ##########################################################
    #############          CLOSING           #################
    ##########################################################
    def close(self):
        print('\n')
        print('----------------------------------------')
        print('       CLOSING THE VISUAL STIMULATION   ')
        print('----------------------------------------')
        self.win.close()
        core.quit()


    ##########################################################
    #############      PRESENTING STIMULI    #################
    ##########################################################

    def run(self, 
            runEvent, readyEvent,
            datafolder, movie_folder,
            dt=10e-3,
            verbose=False):
        """
        we launched the run function
        Once everything is initialized here, it toggles the 'readyEvent' flag
        Then it waits for the 'runEvent' flag to turn on in the interface

        in between the two, stop it by turning off 'readyEvent'
        """

        # We prepare the stimulation
        stim = visual.MovieStim(self.win, 
                                os.path.join(movie_folder,
                                             'movie.mp4'),
                                size=self.screen['resolution'],
                                units='pix')

        self.prepare_stimProps_tables(dt)
        readyEvent.set()

        # waiting for the external trigger [...]
        while not runEvent.is_set():

            if verbose:
                print('waiting for the external trigger [...]')
            time.sleep(0.05)

        ##########################################
        # --> from here external trigger launched  (now runEvent=True)

        ##########################################################
        ###               RUN (while) LOOP                    ####
        ##########################################################

        current_index= -1 # initialize the stimulation index
        print('\n')
        print('--------------------------------------')
        print('     RUNNING VISUAL-STIM PROTOCOL              ')
        print('--------------------------------------\n')

        # we can start the NIdaq recording
        readyEvent.set()

        t0 = time.time()
        stim.play(log=verbose)

        while not os.path.isfile(os.path.join(datafolder.get(), 'NIdaq.start.npy'))\
                and ((time.time()-t0)<3.):
            # we wait for the NIdaq initialisation to be done
            time.sleep(0.05)

        if os.path.isfile(os.path.join(datafolder.get(), 'NIdaq.start.npy')):
            t0 = np.load(os.path.join(datafolder.get(), 'NIdaq.start.npy'))[0]
        else:
            # otherwise 
            t0 = time.time()


        while runEvent.is_set():


            stim.draw()
            self.win.flip()

            t = (time.time()-t0)
            iT = int(t/dt)

            if self.is_interstim[iT] and\
                    (current_index<self.next_index_table[iT]):

                # at each interstim, we re-align the stimulus presentation
                stim.seek(t+0.15) # it takes ~150ms to shift the movie

                # -*- now we update the stimulation display in the terminal -*-
                protocol_id = self.experiment['protocol_id'][self.next_index_table[iT]]
                stim_index = self.experiment['index'][self.next_index_table[iT]]

                # now we update the counter
                current_index = self.next_index_table[iT]

                print(' - t=%.2dh:%.2dm:%.2ds:%.2d' % (t/3600, (t%3600)/60, 
                                                       (t%60), 100*((t%60)-int(t%60))),
                      '- Running protocol of index %i/%i' %\
                            (current_index+1, len(self.experiment['index'])),
                      'protocol #%i, stim #%i' % (protocol_id+1, stim_index+1))

        runEvent.clear() # send the stop signal to all processes

        print('--------------------------------------')
        print(' [ok] protocol terminated successfully')
        print('--------------------------------------')
        self.close()




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
                          vse=False,
                          arrow={'length':20,
                                 'width_factor':0.05,
                                 'color':'red'},
                          with_mask=False):

        """
        """
        tcenter = .5*(self.experiment['time_stop'][episode]-\
                      self.experiment['time_start'][episode])

        
        if self.units in ['cm', 'lin-deg']:
            label={'size':10/self.heights.max()*self.z.max(),
                   'label':'10cm ',
                   'shift_factor':0.02,
                   'lw':1, 'fontsize':10}
        else:
            label={'size':20,'label':'20$^o$  ',
                   'shift_factor':0.02,
                   'lw':1, 'fontsize':10}

        ax = self.show_frame(episode, tcenter, 
                             ax=ax,
                             label=label,
                             with_mask=with_mask)


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

        if ('load_from_protocol_data' in protocol) and\
                            protocol['load_from_protocol_data']:
            # we load a previously saved multiprotocol
            while 'Protocol-%i'%i in protocol:
                subprotocol = {'Screen':protocol['Screen'],
                               'Presentation':'',
                               'demo':(('demo' in protocol) and protocol['demo']),
                               'no-window':True}
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
                    print(' /!\\ "%s" not found in Protocol folder /!\\  ' %\
                                            protocol['Protocol-%i'%i])
                with open(Ppath, 'r') as fp:
                    subprotocol = json.load(fp)
                    subprotocol['Screen'] = protocol['Screen']
                    subprotocol['no-window'] = True
                    subprotocol['demo'] = (('demo' in protocol) and protocol['demo'])
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

        if (protocol['shuffling']=='full'):
            # print('full shuffling of multi-protocol sequence !')
            np.random.seed(protocol['shuffling-seed']) # initializing random seed
            indices = np.arange(len(self.experiment['index']))
            np.random.shuffle(indices)

            for key in self.experiment:
                self.experiment[key] = np.array(self.experiment[key])[indices]

        if (protocol['shuffling']=='per-repeat'):
            # TO BE TESTED
            indices = np.arange(len(self.experiment['index']))
            new_indices = []
            for r in np.unique(self.experiment['repeat']):
                repeat_cond = np.argwhere(self.experiment['repeat']==r).flatten()
                r_indices = indices[repeat_cond]
                np.random.shuffle(r_indices)
                new_indices = np.concatenate([new_indices, r_indices])

            for key in self.experiment:
                self.experiment[key] = np.array(self.experiment[key])[new_indices]

        # we rebuild the time course 
        self.experiment['time_start'][0] =\
                protocol['presentation-prestim-period']
        self.experiment['time_stop'][0] =\
                protocol['presentation-prestim-period']+\
                    self.experiment['time_duration'][0]
        self.experiment['interstim'] = \
                np.concatenate([self.experiment['interstim'][1:],\
                [self.experiment['interstim'][0]]])
        for i in range(1, len(self.experiment['index'])):
            self.experiment['time_start'][i] = \
                    self.experiment['time_stop'][i-1]+\
                    self.experiment['interstim'][i]
            self.experiment['time_stop'][i] = \
                    self.experiment['time_start'][i]+\
                        self.experiment['time_duration'][i]
        self.tstop = self.experiment['time_stop'][-1]+\
                protocol['presentation-poststim-period']
        for key in self.experiment:
            if type(self.experiment[key])==list:
                self.experiment[key] = np.array(self.experiment[key])

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
    return 2*cls.experiment['bg-color'][index]-1.+0.*cls.x

def init_times_frames(cls, index, refresh_freq, security_factor=1.5):
    """ we use this function for each protocol initialisation"""
    interval = cls.experiment['time_stop'][index]-cls.experiment['time_start'][index]
    itend = np.max([1, int(security_factor*interval*refresh_freq)])
    return np.arange(itend), np.arange(itend)/refresh_freq, []

##############################################
## to be used by the multiprocessing module ##
##############################################

def launch_VisualStim(protocol, 
                      runEvent, readyEvent,
                      datafolder, movie_folder):

    demo = ('demo' in protocol) and protocol['demo']
    stim = build_stim(protocol)
    stim.run(runEvent, readyEvent,
             datafolder, movie_folder)


if __name__=='__main__':

    ######################################
    ####  visualize the stimulation   ####
    ######################################

    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("protocol", 
                        help="protocol a json file", 
                        default='')
    parser.add_argument('-s', "--speed", type=float,
                        help="speed to visualize the stimulus (1. by default)", 
                        default=1.)
    parser.add_argument('-t', "--tstop", 
                        type=float, default=15.)
    parser.add_argument("--t0", 
                        help="start time", 
                        default=0.)
    args = parser.parse_args()

    ######################################
    ####     test as a subprocess   ######
    ######################################

    import json, multiprocessing, tempfile
    from ctypes import c_char_p
    from physion.acquisition.tools import base_path
    from physion.visual_stim.build import build_stim

    manager = multiprocessing.Manager() # to share a str across processes
    datafolder = manager.Value(c_char_p, tempfile.gettempdir())
    runEvent = multiprocessing.Event()
    runEvent.clear()
    readyEvent = multiprocessing.Event()
    readyEvent.clear()

    with open(args.protocol, 'r') as fp:
        protocol = json.load(fp)
    protocol['demo'] = True

    movie_folder = \
        os.path.join(os.path.dirname(args.protocol), 'movies',
            os.path.basename(args.protocol.replace('.json','')))

    use_pre_buffering = True
    VisualStim_process = multiprocessing.Process(target=launch_VisualStim,\
            args=(protocol,
                  runEvent, readyEvent,
                  datafolder, movie_folder))
    VisualStim_process.start()

    # -- with use_prebuffering=True
    while not readyEvent.is_set():
        time.sleep(0.1)
    print('\n buffering ready... --> launching stim ! ')
    time.sleep(2)
    runEvent.set()
    time.sleep(args.tstop)
    print(' stoping stim ')
    runEvent.clear()
    


