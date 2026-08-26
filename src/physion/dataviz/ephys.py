import numpy as np
from scipy.ndimage import gaussian_filter1d

import physion.utils.plot_tools as pt
from physion.dataviz import tools as dv_tools

def find_center_channel(data, unit_id):
    return np.argmax(np.std(data.spikeWaveforms[:,:,unit_id],axis=0))

def show_waveforms(data, 
                  unit_id=0,
                  channels_around=5,
                  x_shift_factor=3,
                  y_shift_factor=.4,
                  ax=None,
                  ax_scale=(1.,2.5),
                  color=None):
    """
    deals with the fact that you can have dead channels
    it uses the x,y of good channels to plot the waveforms

    use the shift_factors to move the panels in x and y
    """
    n = find_center_channel(data, unit_id)

    x0 = data.nwbfile.electrodes[n].x[n]
    y0 = data.nwbfile.electrodes[n].y[n]

    if ax is None:
        fig, ax = pt.figure(ax_scale=ax_scale)
    else:
        fig = None
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
        pt.plot(t, wf, ax=ax, no_set=True, color=color)
        pt.annotate(ax, 'ch.%i' % i,
                    ((x-x0)*x_shift_factor,
                        (y-y0)*y_shift_factor),
                    xycoords='data',
                    ha='right', fontsize=4,
                    color=color)

    pt.draw_bar_scales(ax,
                       loc='top-right',
                    Xbar=30, Xbar_label='1ms',
                    Ybar=2, Ybar_label='2$\mu$V',
                    color=color)
    return fig, ax 

def add_spikes(data, tlim, ax,
            fig_fraction_start=0., fig_fraction=1., subsampling=2, color='k',
            scale_side='left',
            name='spikes'):
    pass

def add_MUA(data, tlim, ax,
            fig_fraction_start=0., fig_fraction=1., subsampling=2, color='tab:blue',
            scale_side='left',
            name='MUA'):
    pass

def add_LFP(data, tlim, ax,
            fig_fraction_start=0., fig_fraction=1., 
            vicinity_factor=1.2, 
            smoothing=5,
            subsampling=10, 
            scale=500,
            color=None,
            nTraces=4,
            scale_side='left',
            annotation_side='right',
            name='LFP'):

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['LFP'].data_interfaces['LFP'])

    elecs = data.nwbfile.processing['LFP'].data_interfaces['LFP'].electrodes
    nElec = len(elecs.data[:])
    elecRange = np.linspace(0, nElec, nTraces+1, dtype=int)

    t = dv_tools.convert_index_to_time(range(i1,i2),
        data.nwbfile.processing['LFP'].data_interfaces['LFP'])[::subsampling]

    for n, (e0, e1) in enumerate(zip(elecRange[:-1], elecRange[1:])):

        ypos = n*fig_fraction/nTraces/vicinity_factor+\
                fig_fraction_start # bottom position

        y = gaussian_filter1d(\
            data.nwbfile.processing['LFP'].data_interfaces['LFP'].data[:,e0:e1].mean(axis=-1)[i1:i2][::subsampling],
            smoothing+1e-6)

        rdm = np.random.randint(0, 255) # for color

        dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 500.,
                            ax_fraction_extent=vicinity_factor*fig_fraction/nTraces,
                            ax_fraction_start=ypos,
                            color=color,
                            scale_side=scale_side,
                            scale_unit_string=('%.0f$\\mu$V' if (n==0) else ' '))
        
        # if annotation_side!='':
        #     dv_tools.add_name_annotation(data, ax, 
        #             'roi #%i'%(ir+1), tlim, fig_fraction/len(roiIndices),
        #                                  ypos, color=color, 
        #                                  side=annotation_side)
        
    #     # roi number annotation
    #     roiAnnot = pg.TextItem('%i-' % elecs.data[e0],
    #                             color=(rdm, 255, 255))
    #     roiAnnot.setPos(tt[0], loc+width/nTraces/2.)
    #     self.plot.addItem(roiAnnot)

    # t = dv_tools.convert_index_to_time(range(i1,i2),
    #         data.nwbfile.acquisition['Electrophysiological-Signal'])[::subsampling]

    # y = data.nwbfile.acquisition['Electrophysiological-Signal'].data[i1:i2][::subsampling]

    # dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 0.2,
    #                             ax_fraction_extent=fig_fraction,
    #                             ax_fraction_start=fig_fraction_start,
    #                             scale_side=scale_side,
    #         color=color, scale_unit_string='%.1fmV')

    # dv_tools.add_name_annotation(data, ax, name, tlim,
    #         fig_fraction, fig_fraction_start, color=color)

