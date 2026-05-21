import numpy as np
import physion.utils.plot_tools as pt

def find_center_channel(data, unit_id):
    return np.argmax(np.std(data.spikeWaveforms[:,:,unit_id],axis=0))

def show_waveforms(data, 
                  unit_id=0,
                  channels_around=5,
                  x_shift_factor=3,
                  y_shift_factor=.4,
                  ax_scale=(1.,2.5)):
    """
    deals with the fact that you can have dead channels
    it uses the x,y of good channels to plot the waveforms

    use the shift_factors to move the panels in x and y
    """
    n = find_center_channel(data, unit_id)

    x0 = data.nwbfile.electrodes[n].x[n]
    y0 = data.nwbfile.electrodes[n].y[n]

    fig, ax = pt.figure(ax_scale=ax_scale)
    ax.axis('off')
    for i in np.clip(\
            np.arange(n-channels_around, n+channels_around-1),
            0, len(data.nwbfile.electrodes)-1):

        x = data.nwbfile.electrodes[i].x[i]
        y = data.nwbfile.electrodes[i].y[i]

        t = (x-x0)*x_shift_factor+\
            np.arange(data.spikeWaveforms.shape[0])
        wf = data.spikeWaveforms[:,i,unit_id]+\
            (y-y0)*y_shift_factor
        pt.plot(t, wf, ax=ax, no_set=True)
        pt.annotate(ax, 'ch.%i' % i,
                    ((x-x0)*x_shift_factor,
                        (y-y0)*y_shift_factor),
                    xycoords='data',
                    ha='right', fontsize=4)

    pt.draw_bar_scales(ax,
                       loc='top-right',
                    Xbar=30, Xbar_label='1ms',
                    Ybar=2, Ybar_label='2$\mu$V')
    return fig, ax 
