import numpy as np
import itertools, os, sys, pathlib, time, json, tempfile
try:
    from psychopy import visual, core, event, clock, monitors
except ModuleNotFoundError:
    pass

from physion.visual_stim.screens import SCREENS
from physion.visual_stim.stimuli import *


def build_stim(protocol):
    """
    """

    if (protocol['Presentation']=='multiprotocol'):
        return multiprotocol(protocol)
    else:
        protocol_name = protocol['Stimulus'].replace('-image','').replace('-', '_').replace('+', '_')
        if hasattr(sys.modules[__name__], protocol_name):
            return getattr(sys.modules[__name__], protocol_name)(protocol) # e.g. returns "center_grating_image_stim(protocol)"
        else:
            print(protocol_name)
            print(protocol)
            print('\n /!\ Protocol not recognized ! /!\ \n ')
            return None


def stop_signal(parent):
    if (len(event.getKeys())>0) or (parent.stop_flag):
        parent.stop_flag = True
        if hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation stopped !')
        return True
    else:
        return False


class visual_stim:

    def __init__(self,
                 protocol,
                 demo=False):
        """
        """
        self.protocol = protocol
        self.screen = SCREENS[self.protocol['Screen']]
        self.buffer = None # by default, non-buffered data
        self.buffer_delay = 0            

        self.protocol['movie_refresh_freq'] = protocol['movie_refresh_freq'] if 'movie_refresh_freq' in protocol else 10.

        if demo or (('demo' in self.protocol) and self.protocol['demo']):
            # --------------------- #
            ## ---- DEMO MODE ---- ##    we override the parameters
            # --------------------- #
            self.screen['monitoring_square']['size'] = int(600*self.screen['monitoring_square']['size']/self.screen['resolution'][0])
            self.screen['resolution'] = (800,int(800*self.screen['resolution'][1]/self.screen['resolution'][0]))
            self.screen['screen_id'] = 0
            self.screen['fullscreen'] = False

        # then we can initialize the angle
        self.set_angle_meshgrid()

        if not ('no-window' in self.protocol):

            self.monitor = monitors.Monitor(self.screen['name'])
            self.monitor.setDistance(self.screen['distance_from_eye'])
            self.k, self.gamma = self.screen['gamma_correction']['k'], self.screen['gamma_correction']['gamma']

            self.win = visual.Window(self.screen['resolution'], monitor=self.monitor,
                                     screen=self.screen['screen_id'], fullscr=self.screen['fullscreen'],
                                     units='pix',
                                     color=self.gamma_corrected_lum(self.protocol['presentation-prestim-screen']))

            # ---- blank screens ----

            self.blank_start = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                  color=self.gamma_corrected_lum(self.protocol['presentation-prestim-screen']),
                                                  units='pix')
            if 'presentation-interstim-screen' in self.protocol:
                self.blank_inter = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                      color=self.gamma_corrected_lum(self.protocol['presentation-interstim-screen']),
                                                      units='pix')
            self.blank_end = visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                                                color=self.gamma_corrected_lum(self.protocol['presentation-poststim-screen']),
                                                units='pix')


            # ---- monitoring square properties ----

            if self.screen['monitoring_square']['location']=='top-right':
                pos = [int(x/2.-self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='bottom-left':
                pos = [int(-x/2.+self.screen['monitoring_square']['size']/2.) for x in self.screen['resolution']]
            elif self.screen['monitoring_square']['location']=='top-left':
                pos = [int(-self.screen['resolution'][0]/2.+self.screen['monitoring_square']['size']/2.),
                       int(self.screen['resolution'][1]/2.-self.screen['monitoring_square']['size']/2.)]
            elif self.screen['monitoring_square']['location']=='bottom-right':
                pos = [int(self.screen['resolution'][0]/2.-self.screen['monitoring_square']['size']/2.),
                       int(-self.screen['resolution'][1]/2.+self.screen['monitoring_square']['size']/2.)]
            else:
                print(30*'-'+'\n /!\ monitoring square location not recognized !!')

            self.on = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'], pos=pos, sf=0,
                                         color=self.screen['monitoring_square']['color-on'], units='pix')
            self.off = visual.GratingStim(win=self.win, size=self.screen['monitoring_square']['size'],  pos=pos, sf=0,
                                          color=self.screen['monitoring_square']['color-off'], units='pix')

            # initialize the times for the monitoring signals
            self.Ton = int(1e3*self.screen['monitoring_square']['time-on'])
            self.Toff = int(1e3*self.screen['monitoring_square']['time-off'])
            self.Tfull, self.Tfull_first = int(self.Ton+self.Toff), int((self.Ton+self.Toff)/2.)


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
        #  ------- for simplicity -------  #
        # we linearize the arctan function #
        """
        dAngle_per_pix = self.pix_to_angle(1.)
        x, z = np.meshgrid(dAngle_per_pix*(np.arange(self.screen['resolution'][0])-self.screen['resolution'][0]/2.),
                           dAngle_per_pix*(np.arange(self.screen['resolution'][1])-self.screen['resolution'][1]/2.),
                           indexing='xy')
        self.x, self.z = x.T, z.T

    def angle_to_pix(self, angle):
        # using the above linear approx, the relationship is just the inverse:
        return angle/self.pix_to_angle(1.)

    # some general grating functions
    def compute_rotated_coords(self, angle,
                               xcenter=0, zcenter=0):
        return (self.x-xcenter)*np.cos(angle/180.*np.pi)+(self.z-zcenter)*np.sin(angle/180.*np.pi)

    def compute_grating(self, xrot,
                        spatial_freq=0.1, contrast=1, time_phase=0.):
        return contrast*(1+np.cos(2*np.pi*(spatial_freq*xrot-time_phase)))/2.

    ################################
    #  ---     Experiment      --- #
    ################################

    def init_experiment(self, protocol, keys, run_type='static'):

        self.experiment, self.PATTERNS = {}, []

        if protocol['Presentation']=='Single-Stimulus':
            # single stimulus type
            for key in protocol:
                if key.split(' (')[0] in keys:
                    self.experiment[key.split(' (')[0]] = [protocol[key]]
                    self.experiment['index'] = [0]
                    self.experiment['frame_run_type'] = [run_type]
                    self.experiment['index'] = [0]
                    self.experiment['time_start'] = [protocol['presentation-prestim-period']]
                    self.experiment['time_stop'] = [protocol['presentation-duration']+protocol['presentation-prestim-period']]
                    self.experiment['time_duration'] = [protocol['presentation-duration']]
                    self.experiment['interstim'] = [protocol['presentation-interstim-period'] if 'presentation-interstim-period' in protocol else 0]
                    self.experiment['interstim-screen'] = [protocol['presentation-interstim-screen'] if 'presentation-interstim-screen' in protocol else 0]

        else:
            # ------------  MULTIPLE STIMS ------------
            VECS, FULL_VECS = [], {}
            for key in keys:
                FULL_VECS[key], self.experiment[key] = [], []
                if ('N-log-'+key in protocol) and (protocol['N-log-'+key]>1):
                    VECS.append(np.logspace(np.log10(protocol[key+'-1']), np.log10(protocol[key+'-2']),protocol['N-log-'+key]))
                elif protocol['N-'+key]>1:
                    VECS.append(np.linspace(protocol[key+'-1'], protocol[key+'-2'],protocol['N-'+key]))
                else:
                    VECS.append(np.array([protocol[key+'-2']])) # we pick the SECOND VALUE as the constant one (so remember to fill this right in GUI)
            for vec in itertools.product(*VECS):
                for i, key in enumerate(keys):
                    FULL_VECS[key].append(vec[i])

            for k in ['index', 'repeat','time_start', 'time_stop',
                    'interstim', 'time_duration', 'interstim-screen', 'frame_run_type']:
                self.experiment[k] = []

            index_no_repeat = np.arange(len(FULL_VECS[key]))

            # then dealing with repetitions
            Nrepeats = max([1,protocol['N-repeat']])

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
                    self.experiment['repeat'].append(r)
                    self.experiment['time_start'].append(protocol['presentation-prestim-period']+\
                                                         (r*len(index_no_repeat)+n)*\
                                                         (protocol['presentation-duration']+protocol['presentation-interstim-period']))
                    self.experiment['time_stop'].append(self.experiment['time_start'][-1]+protocol['presentation-duration'])
                    self.experiment['interstim'].append(protocol['presentation-interstim-period'])
                    self.experiment['interstim-screen'].append(protocol['presentation-interstim-screen'])
                    self.experiment['time_duration'].append(protocol['presentation-duration'])
                    self.experiment['frame_run_type'].append(run_type)

        for k in ['index', 'repeat','time_start', 'time_stop',
                    'interstim', 'time_duration', 'interstim-screen', 'frame_run_type']:
            self.experiment[k] = np.array(self.experiment[k]) 

    # the close function
    def close(self):
        self.win.close()

    def quit(self):
        core.quit()

    # screen at start
    def start_screen(self, parent):
        if not parent.stop_flag:
            self.blank_start.draw()
            self.off.draw()
            try:
                self.win.flip()
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-prestim-period'])

    # screen at end
    def end_screen(self, parent):
        if not parent.stop_flag:
            self.blank_end.draw()
            self.off.draw()
            try:
                self.win.flip()
            except AttributeError:
                pass
            clock.wait(self.protocol['presentation-poststim-period'])

    # screen for interstim
    def inter_screen(self, parent, duration=1., color=0):
        if not parent.stop_flag and hasattr(self, 'blank_inter') and duration>0:
            visual.GratingStim(win=self.win, size=10000, pos=[0,0], sf=0,
                               color=self.gamma_corrected_lum(color), units='pix').draw()
            self.off.draw()
            try:
                self.win.flip()
            except AttributeError:
                pass
            clock.wait(duration)

    # blinking in one corner
    def add_monitoring_signal(self, new_t, start):
        """ Pulses of length Ton at the times : [0, 0.5, 1, 2, 3, 4, ...] """
        if (int(1e3*new_t-1e3*start)<self.Tfull) and (int(1e3*new_t-1e3*start)%self.Tfull_first<self.Ton):
            self.on.draw()
        elif int(1e3*new_t-1e3*start)%self.Tfull<self.Ton:
            self.on.draw()
        else:
            self.off.draw()

    def add_monitoring_signal_sp(self, new_t, start):
        """ Single pulse monitoring signal (see array_run) """
        if (int(1e3*new_t-1e3*start)<self.Ton):
            self.on.draw()
        else:
            self.off.draw()


    ##########################################################
    #############      PRESENTING STIMULI    #################
    ##########################################################


    #####################################################
    # adding a run purely define by an array (time, x, y), see e.g. sparse_noise initialization
    def array_sequence_presentation(self, parent, index):
        tic = time.time()
        # print('stim_index', self.experiment['index'][index])
        # -------------------------------------------------------
        time_indices, frames, refresh_freq = self.get_frames_sequence(index) # refresh_freq can be stimulus dependent !
        print('  array init took %.1fs' % (time.time()-tic))
        toc = time.time()
        FRAMES = []
        for frame in frames:
            FRAMES.append(visual.ImageStim(self.win,
                                           image=self.gamma_corrected_lum(frame),
                                           units='pix', size=self.win.size))
        print('  array buffering took %.1fs' % (time.time()-toc))
        print('  full episode init took %.1fs' % (time.time()-tic))
        self.buffer_delay = np.max([self.buffer_delay, time.time()-tic])
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*refresh_freq) # refresh_freq can be stimulus dependent !
            if iframe>=len(time_indices):
                print('for index:')
                print(iframe, len(time_indices), len(frames), refresh_freq)
                print(' /!\ Pb with time indices index  /!\ ')
                print('forcing lower values')
                iframe = len(time_indices)-1
            FRAMES[time_indices[iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    #####################################################
    # adding a run purely define by an array -- NOW BUFFERED 
    def buffer_stim(self, parent, gui_refresh_func=None):
        """
        we build the buffers order so that we can call them as:
        self.buffer[protocol_index][stim_index] 
        where:
        protocol_index = stim.experiment['protocol_id'][index]
        stim_index = stim.experiment['index'][index]
        where "index" is the episode number over the full protocol run (including multiprotocols)
        """
        cls = (parent if parent is not None else self)
        win = cls.win if hasattr(cls, 'win') else self.win

        self.buffer = []
        if 'protocol_id' in self.experiment:
            protocol_ids = self.experiment['protocol_id']
        else:
            protocol_ids = np.zeros(len(self.experiment['index']), dtype=int)

        print(' --> buffering stimuli [...] ') 
        tic = time.time()
        for protocol_id in np.sort(np.unique(protocol_ids)):
            self.buffer.append([]) # adding a new set of buffers
            print('    - protocol %i  ' % (protocol_id+1)) 
            single_indices = np.arange(len(protocol_ids))[(protocol_ids==protocol_id) & (self.experiment['repeat']==0)] # this gives the valid
            indices_order = np.argsort(self.experiment['index'][single_indices])
            for stim_index, index_in_full_array in enumerate(single_indices[indices_order]):
                toc = time.time()
                time_indices, frames, refresh_freq = self.get_frames_sequence(index_in_full_array)
                self.buffer[protocol_id].append({'time_indices':time_indices,
                                                 'frames':frames,
                                                 'FRAMES':[],
                                                 'refresh_freq':refresh_freq})
                for frame in self.buffer[protocol_id][stim_index]['frames']:
                    self.buffer[protocol_id][stim_index]['FRAMES'].append(visual.ImageStim(win,
                                                                          image=self.gamma_corrected_lum(frame),
                                                                          units='pix', size=win.size))
                    if gui_refresh_func is not None:    
                        gui_refresh_func()
                print('        index #%i   (%.2fs)' % (stim_index+1, time.time()-toc)) 
   
        print(' --> buffering done ! (t=%.2fs / %.2fmin)' % (time.time()-tic, (time.time()-tic)/60.)) 
        return True

    def array_sequence_buffered_presentation(self, parent, index):
        # --- fetch protocol_id and stim_index:
        protocol_id = self.experiment['protocol_id'][index] if 'protocol_id' in self.experiment else 0
        stim_index = self.experiment['index'][index]
        # print('stim_index', stim_index)
        # -------------------------------------------------------
        # then run loop over buffered frames
        start = clock.getTime()
        while ((clock.getTime()-start)<(self.experiment['time_duration'][index])) and not parent.stop_flag:
            iframe = int((clock.getTime()-start)*self.buffer[protocol_id][stim_index]['refresh_freq'])
            #print(self.buffer[protocol_id][stim_index]['refresh_freq'])
            #print(iframe, len(self.buffer[protocol_id][stim_index]['time_indices']))
            self.buffer[protocol_id][stim_index]['FRAMES'][self.buffer[protocol_id][stim_index]['time_indices'][iframe]].draw()
            self.add_monitoring_signal(clock.getTime(), start)
            try:
                self.win.flip()
            except AttributeError:
                pass


    ## FINAL RUN FUNCTION
    def run(self, parent=None):

        try:
            t0 = np.load(os.path.join(str(parent.datafolder.get()), 'NIdaq.start.npy'))[0]
        except FileNotFoundError:
            print(str(parent.datafolder.get()), 'NIdaq.start.npy', 'not found !')
            t0 = time.time()

        self.start_screen(parent) 

        if ('buffer' in self.protocol) and self.protocol['buffer'] and (self.buffer is None):
            self.buffer_stim(parent)

        for i in range(len(self.experiment['index'])):
            if stop_signal(parent):
                break
            t = time.time()-t0
            print('t=%.2dh:%.2dm:%.2fs - Running protocol of index %i/%i                                protocol-ID:%i' % (t/3600,
                (t%3600)/60, (t%60), i+1, len(self.experiment['index']),
                 self.experiment['protocol_id'][i] if 'protocol_id' in self.experiment else 0))

            # ---- single_episode_run ----- #
            if self.buffer is not None:
                self.array_sequence_buffered_presentation(parent, i) # buffered version
            else: # non-buffered by defaults
                self.array_sequence_presentation(parent, i) # non-buffered version

            if i<(len(self.experiment['index'])-1):
                self.inter_screen(parent,
                                  duration=1.*self.experiment['interstim'][i],
                                  color=self.experiment['interstim-screen'][i])
        self.end_screen(parent)
        if not parent.stop_flag and hasattr(parent, 'statusBar'):
            parent.statusBar.showMessage('stimulation over !')


    ##########################################################
    #############    DRAWING STIMULI (offline)  ##############
    ##########################################################

    def get_image(self, episode, time_from_episode_start=0, parent=None):
        print('to be implemented in child class')
        return 0*self.x

    def plot_stim_picture(self, episode,
                          ax=None, parent=None,
                          label=None, vse=False):

        cls = (parent if parent is not None else self)
        ax = self.show_frame(episode,
                             ax=ax,
                             label=label,
                             vse=vse,
                             parent=parent)

        return ax

    def get_prestim_image(self):
        return (1+self.protocol['presentation-prestim-screen'])/2.+0*self.x
    def get_interstim_image(self):
        return (1+self.protocol['presentation-interstim-screen'])/2.+0*self.x
    def get_poststim_image(self):
        return (1+self.protocol['presentation-poststim-screen'])/2.+0*self.x

    def image_to_frame(self, img, norm=False, psychopy_to_numpy=False):
        """ need to transpose given the current coordinate system"""
        if psychopy_to_numpy:
            return img.T/2.+0.5
        if norm:
            return (img.T-np.min(img))/(np.max(img)-np.min(img)+1e-6)
        else:
            return img.T

    def get_vse(self, episode, parent=None):
        """
        Virtual Scene Exploration dictionary
        None by default, should be overriden by method in children class
        """
        return None

    def show_frame(self, episode,
                   time_from_episode_start=0,
                   parent=None,
                   label={'degree':5,
                          'shift_factor':0.02,
                          'lw':2, 'fontsize':12},
                   arrow=None,
                   vse=False,
                   ax=None,
                   return_img=False):
        """

        display the visual stimulus at a given time in a given episode of a stimulation pattern

        --> optional with angular label (switch to None to remove)
                   label={'degree':5,
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
            fig, ax = plt.subplots(1)

        cls = (parent if parent is not None else self)
        
        try:
            img = ax.imshow(cls.image_to_frame(cls.get_image(episode,
                                                       time_from_episode_start=time_from_episode_start,
                                                       parent=cls), psychopy_to_numpy=True),
                            extent=(0, self.screen['resolution'][0], 0, self.screen['resolution'][1]),
                      cmap='gray', vmin=0, vmax=1,
                      origin='lower',
                      aspect='equal')
        except BaseException as be:
            print(be)
            print(' /!\ pb in get stim /!\  ')

        if vse:
            self.vse = self.get_vse(episode, parent=cls)
            if self.vse is not None:
                self.add_vse(ax, self.vse)

        ax.axis('off')

        if label is not None:
            nz, nx = self.x.shape
            L, shift = nx/(self.x[-1][-1]-self.x[0][0])*label['degree'], label['shift_factor']*nx
            ax.plot([-shift, -shift], [-shift,L-shift], 'k-', lw=label['lw'])
            ax.plot([-shift, L-shift], [-shift,-shift], 'k-', lw=label['lw'])
            ax.annotate('%.0f$^o$ ' % label['degree'], (-shift, -shift), fontsize=label['fontsize'], ha='right', va='bottom')

        if return_img:
            return img
        else:
            return ax

    def update_frame(self, episode, img,
                     time_from_episode_start=0,
                     parent=None):
        cls = (parent if parent is not None else self)
        
        img.set_array(cls.image_to_frame(cls.get_image(episode,
                                                      time_from_episode_start=time_from_episode_start,
                                                     parent=cls), psychopy_to_numpy=True))


    def add_arrow(self, arrow, ax):
        nz, nx = self.x.shape
        ax.arrow(self.angle_to_pix(arrow['center'][0])+nx/2,
                 self.angle_to_pix(arrow['center'][1])+nz/2,
                 np.cos(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 -np.sin(np.pi/180.*arrow['direction'])*self.angle_to_pix(arrow['length']),
                 width=self.angle_to_pix(arrow['length'])*arrow['width_factor'],
                 color=arrow['color'])

    def add_vse(self, ax, vse):
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.plot([self.screen['resolution'][0]/2.]+list(vse['x']),
                [self.screen['resolution'][1]/2.]+list(vse['y']), 'o-', color='#d62728', lw=0.5, ms=2)


#####################################################
##  ----      MOVIE STIMULATION REPLAY      --- #####
#####################################################

class movie_replay(visual_stim):
    """ TO BE IMPLEMENTED """

    def __init__(self, protocol):

        super().__init__(protocol)

    def run(self, parent):
        pass



