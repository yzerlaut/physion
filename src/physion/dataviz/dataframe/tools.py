import numpy as np
import matplotlib.pylab as plt

def add_name_annotation(ax,
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

def shifted_stop(tlim, frac_shift=0.01):
    return tlim[1]+frac_shift*(tlim[1]-tlim[0])


def plot_scaled_signal(dataframe, 
                       ax, label,
                       tlim, 
                       scale_bar=1,
                       scale_side='left', 
                       scale_label='1 s.d.',
                       ax_fraction_extent=1, ax_fraction_start=0,
                       color='#1f77b4'):
    """
    # generic function to add scaled signal
    """

    tcond = (dataframe['time']>=tlim[0]) & (dataframe['time']<=tlim[1])
    t = dataframe['time'][tcond]

    signal = dataframe[label] # signal without zoom for scale
    try:
        scale_range = np.max([signal.max()-signal.min(), 1.1*scale_bar])
        min_signal = signal.min()
    except ValueError:
        scale_range = scale_bar
        min_signal = 0
    signal = signal[tcond] # with zoom for plot

    ax.plot(t,
            ax_fraction_start+(signal-min_signal)*ax_fraction_extent/scale_range,
            color=color, lw=1)

    if scale_side=='left':
        tscale, side = shifted_start(tlim), 'right'
    else:
        tscale, side = shifted_stop(tlim), 'left'

    # add scale bar
    if scale_side!='':
        ax.plot(tscale*np.ones(2),
                ax_fraction_start+scale_bar*np.arange(2)*ax_fraction_extent/scale_range,
                color=color, lw=1)
        ax.annotate(scale_label,
                (tscale, ax_fraction_start),
                ha=side, color=color, xycoords='data', fontsize=7)

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

    

