import os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt
plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parents[1],\
                'utils', 'matplotlib_style.py'))

#############################################
##           Matplotlib Display      ########
#############################################

def add_name_annotation(data,
                        ax,
                        name,
                        tlim, ax_fraction_extent, ax_fraction_start,
                        color='k', rotation=0, side='right'):
    if side=='right':
        ax.annotate(' '+name,
                (tlim[1], ax_fraction_extent/2.+ax_fraction_start),
                xycoords='data', color=color, va='center', rotation=rotation)
    else:
        ax.annotate(name+' ',
                (tlim[0], ax_fraction_extent/2.+ax_fraction_start),
                xycoords='data', color=color, va='center', ha='right', rotation=rotation)


def shifted_start(tlim, frac_shift=0.01):
    return tlim[0]-frac_shift*(tlim[1]-tlim[0])


def plot_scaled_signal(data, 
                       ax, t, signal,
                       tlim, scale_bar,
                       ax_fraction_extent=1, ax_fraction_start=0,
                       color='#1f77b4', scale_unit_string='%.1f'):
    """
    # generic function to add scaled signal
    """

    try:
        scale_range = np.max([signal.max()-signal.min(), scale_bar])
        min_signal = signal.min()
    except ValueError:
        scale_range = scale_bar
        min_signal = 0

    ax.plot(t,
            ax_fraction_start+(signal-min_signal)*ax_fraction_extent/scale_range,
            color=color, lw=1)

    # add scale bar
    if scale_unit_string!='':
        ax.plot(shifted_start(tlim)*np.ones(2),
                ax_fraction_start+scale_bar*np.arange(2)*ax_fraction_extent/scale_range,
                color=color, lw=1)

    # add annotation
    if '%' in scale_unit_string:
        ax.annotate(str(scale_unit_string+' ') % scale_bar,
                (shifted_start(tlim), ax_fraction_start),
                ha='right', color=color, va='center', xycoords='data')
    elif scale_unit_string!='':
        ax.annotate(scale_unit_string,
                (shifted_start(tlim), ax_fraction_start),
                ha='right', color=color, va='center', xycoords='data')


def add_bar_annotations(ax,
                        Xbar=0, Xbar_label='',
                        Ybar=0, Ybar_label='',
                        lw=2, fontsize=10):

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(xlim[0]+Xbar*np.arange(2), ylim[1]*np.ones(2), 'k-', lw=lw)
    ax.annotate(Xbar_label, (xlim[0], ylim[1]), fontsize=fontsize)
    ax.plot(xlim[0]*np.ones(2), ylim[1]-Ybar*np.arange(2), 'k-', lw=lw)
    ax.annotate(Ybar_label, (xlim[0], ylim[1]), 
            fontsize=fontsize, ha='right', va='top', rotation=90)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    
#######################################
##           Pyqt Display      ########
#######################################

def scale_and_position(self, y,
                       value=None, 
                       iHeight=1):
    

    if value is None:
        value = y

    ymin, ymax = y.min(), y.max()

    if ymin<ymax:
        y = self.iplot+iHeight*(value-ymin)/(ymax-ymin)
    else:
        y = self.iplot+iHeight*value

    self.iplot += iHeight

    return y


def shift(self, i):
    return settings['blank-space']*i+\
        np.sum(np.power(settings['increase-factor'], np.arange(i)))


def convert_time_to_index(time, nwb_quantity, axis=0):
    if nwb_quantity.timestamps is not None:
        cond = nwb_quantity.timestamps[:]>=time
        if np.sum(cond)>0:
            return np.arange(nwb_quantity.timestamps.shape[0])[cond][0]
        else:
            return nwb_quantity.timestamps.shape[0]-1
    elif nwb_quantity.starting_time is not None:
        t = time-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[axis]-1 # maybe shift to -1 to handle images
        return max([1, min([int(t/dt), imax-1])]) # then we add +1 / -1 in the visualization
    else:
        return 0


def convert_times_to_indices(t1, t2, nwb_quantity, axis=0):
    if nwb_quantity.timestamps is not None:
        cond = (nwb_quantity.timestamps[:]>=t1) & (nwb_quantity.timestamps[:]<=t2)
        if np.sum(cond)>0:
            return np.arange(nwb_quantity.timestamps.shape[0])[cond][np.array([0,-1])]
        else:
            return (0, nwb_quantity.timestamps.shape[axis]-1)
    elif nwb_quantity.starting_time is not None:
        T1, T2 = t1-nwb_quantity.starting_time, t2-nwb_quantity.starting_time
        dt = 1./nwb_quantity.rate
        imax = nwb_quantity.data.shape[axis]-1 # maybe shift to -1 to handle images
        return (max([1, min([int(T1/dt), imax-1])]), max([1, min([int(T2/dt), imax-1])]))
    else:
        return (0, imax)


def extract_from_times(t1, t2, nwb_quantity, axis=0):
    
    imax = nwb_quantity.data.shape[axis]-1
    
    if nwb_quantity.timestamps is not None:
        
        cond = (nwb_quantity.timestamps[:]>=t1) & (nwb_quantity.timestamps[:]<=t2)
        if np.sum(cond)>0:
            indices = np.arange(nwb_quantity.timestamps.shape[axis])[cond]
            times = nwb_quantity.timestamps[cond]
        else:
            ii, indices = np.argmin((nwb_quantity.timestamps[:]-t1)**2), [ii]
            times = [nwb_quantity.timestamps[ii]]
            
    elif nwb_quantity.starting_time is not None:
        
        dt = 1./nwb_quantity.rate
        i1, i2 = int((t1-nwb_quantity.starting_time)/dt), int((t2-nwb_quantity.starting_time)/dt)
        indices = np.arange(imax+1)[max([0, min([i1, imax])]):max([0, min([i2, imax])])]
        times = nwb_quantity.starting_time+dt*indices
        
    else:
        
        indices = [0]
        times = [0]
    
    return indices, times
    

def convert_index_to_time(index, nwb_quantity):
    """ index can be an array """
    if nwb_quantity.timestamps is not None:
        return nwb_quantity.timestamps[index]
    else:
        return nwb_quantity.starting_time+index/nwb_quantity.rate



#############################################
##             Others                ########
#############################################

def format_key_value(key, value):
    if key in ['angle','direction']:
        return '$\\theta$=%.0f$^{o}$' % value
    elif key=='x-center':
        return '$x$=%.0f$^{o}$' % value
    elif key=='y-center':
        return '$y$=%.0f$^{o}$' % value
    elif key=='radius':
        return '$r$=%.0f$^{o}$' % value
    elif key=='size':
        return '$s$=%.0f$^{o}$' % value
    elif key=='contrast':
        return '$c$=%.2f' % value 
    elif key=='repeat':
        return 'trial #%i' % (value+1)
    elif key=='center-time':
        return '$t_0$:%.1fs' % value
    elif key=='Image-ID':
        return 'im#%i' % value
    elif key=='VSE-seed':
        return 'vse#%i' % value
    elif key=='light-level':
        if value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'lum.=%.1f' % value
    elif key=='dotcolor':
        if value==-1:
            return 'black dot'
        elif value==0:
            return 'grey dot'
        elif value==1:
            return 'white dot'
        else:
            return 'dot=%.1f' % value
    elif key=='color':
        if value==-1:
            return 'black'
        elif value==0:
            return 'grey'
        elif value==1:
            return 'white'
        else:
            return 'color=%.1f' % value
    elif key=='speed':
        return 'v=%.0f$^{o}$/s' % value
    elif key=='protocol_id':
        return 'p.#%i' % (value+1)
    else:
        return '%s=%.2f' % (key, value)


settings = {
    'window_size':(1000,600),
    # raw data plot settings
    # so "Calcium" is twice "Electrophy", that is twice "Pupil",..  "Locomotion"
    'increase-factor':2, 
    'blank-space':0.1, 
    'colors':{'Screen':(100, 100, 100, 255),#'grey',
              'Locomotion':(255,255,255,255),#'white',
              'FaceMotion':(255,0,255,255),#'purple',
              'Pupil':(255,0,0,255),#'red',
              'Gaze':(200,100,0,255),#'orange',
              'Electrophy':(100,100,255,255),#'blue',
              'LFP':(100,100,255,255),#'blue',
              'Vm':(100,100,100,255),#'blue',
              'CaImaging':(0,255,0,255)},#'green'},
    # general settings
    'Npoints':500}


