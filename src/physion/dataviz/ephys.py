import numpy as np
from scipy.ndimage import gaussian_filter1d

import physion.utils.plot_tools as pt
from physion.dataviz import tools as dv_tools

def contact_position(data, i):
    """ position of the contact on the probe (um), "x", "y" in former files """
    elecs = data.nwbfile.electrodes
    if 'rel_y' in elecs.colnames:
        return elecs['rel_x'][i], elecs['rel_y'][i]
    else:
        return elecs['x'][i], elecs['y'][i]

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

    x0, y0 = contact_position(data, n)

    if ax is None:
        fig, ax = pt.figure(ax_scale=ax_scale)
    else:
        fig = None
    ax.axis('off')
    for i in np.clip(\
            np.arange(n-channels_around, n+channels_around-1),
            0, len(data.nwbfile.electrodes)-1):

        x, y = contact_position(data, i)

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
                    Ybar=2, Ybar_label='2$\\mu$V',
                    color=color)
    return fig, ax 

def add_spikes(data, tlim, ax,
            fig_fraction_start=0., fig_fraction=1., color='k',
            ms=0.2,
            name='spikes'):
    """
    raster of the spikes of the selected units (data.selected_units),
        units ordered by depth (top of the probe at the top)
        if the depths are available (spike waveforms), by unit index otherwise

    the raster is rasterized (for light vector graphics with many spikes)
    """
    units = data.selected_units

    if not hasattr(data, 'depth_units'):
        data.build_depth_units()
    if data.depth_units is not None:
        order = np.argsort(np.argsort(data.depth_units)) # rank of the depth
    else:
        order = np.arange(len(units))

    T, Y = [], []
    for i, unit in enumerate(units):
        spikes = np.array(data.nwbfile.units['spike_times'][unit])
        spikes = spikes[(spikes>=tlim[0]) & (spikes<=tlim[1])]
        T.append(spikes)
        # top unit (shallowest) at the top
        Y.append(np.ones(len(spikes))*(len(units)-1-order[i]))
    T, Y = np.concatenate(T), np.concatenate(Y)

    ax.plot(T, fig_fraction_start+(Y+.5)*fig_fraction/len(units),
            'o', ms=ms, mew=0, color=color, rasterized=True)

    ax.annotate('%i units' % len(units),
                (dv_tools.shifted_start(tlim), fig_fraction_start+fig_fraction/2.),
                xycoords='data', color=color, va='center', ha='right',
                rotation=90, fontsize=7)

    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_electrical_series(data, key, tlim, ax,
                          fig_fraction_start=0., fig_fraction=1., 
                          vicinity_factor=1.2, 
                          smoothing=5,
                          subsampling=2, 
                          scale=20,
                          color='tab:blue',
                          nTraces=4,
                          scale_side='left',
                          name=None):
    """
    electrical series "key" ("LFP" or "MUA") of the selected channels, 
        averaged over "nTraces" groups of neighboring channels 
        (ordered by depth, top of the probe at the top)

    smoothing: Gaussian smoothing (in samples, after subsampling)
    scale: scale bar (in uV)
    """
    series = data.nwbfile.processing[key].data_interfaces[key]

    i1, i2 = dv_tools.convert_times_to_indices(*tlim, series)

    t = dv_tools.convert_index_to_time(range(i1,i2), series)[::subsampling]

    # channels of the selection, ordered from the top to the bottom of the probe
    indices, rows = data.selected_series_channels(key)
    depths = data.depth_of_electrodes(rows, key)
    if depths is not None:
        indices, depths = indices[np.argsort(depths)], np.sort(depths)

    # subsampled reading of all the channels (h5 file), then the selected ones
    signals = series.data[i1:i2:subsampling][:,indices]
    groups = np.array_split(np.arange(len(indices)), nTraces)

    for n, group in enumerate(groups):

        # first group (top) at the top
        ypos = (nTraces-1-n)*fig_fraction/nTraces+\
                fig_fraction_start # bottom position

        y = gaussian_filter1d(signals[:,group].mean(axis=-1),
                              smoothing+1e-6)

        dv_tools.plot_scaled_signal(data, ax, t, y, tlim, scale,
                            ax_fraction_extent=vicinity_factor*fig_fraction/nTraces,
                            ax_fraction_start=ypos,
                            color=color,
                            scale_side=scale_side,
                            scale_unit_string=('%.0f$\\mu$V' if (n==nTraces-1) else ' '))

        # depth range of the group (or channel rows)
        if depths is not None:
            label = '%.0f-%.0f$\\mu$m' % (depths[group[0]], depths[group[-1]])
        else:
            label = 'ch.%i-%i' % (rows[group[0]], rows[group[-1]])
        ax.annotate(' '+label, (tlim[1], ypos+fig_fraction/nTraces/2.),
                    xycoords='data', color=color, va='center', fontsize=6)

    dv_tools.add_name_annotation(data, ax, key if name is None else name, 
            tlim, fig_fraction, fig_fraction_start, color=color, 
            rotation=90, side='left')


def add_MUA(data, tlim, ax,
            fig_fraction_start=0., fig_fraction=1., 
            vicinity_factor=1.2, 
            smoothing=5,
            subsampling=2, 
            scale=20,
            color='tab:blue',
            nTraces=4,
            scale_side='left',
            name='MUA'):
    """ MUA in depth bands (see add_electrical_series) """
    add_electrical_series(data, 'MUA', tlim, ax,
                          fig_fraction_start=fig_fraction_start, 
                          fig_fraction=fig_fraction, 
                          vicinity_factor=vicinity_factor, 
                          smoothing=smoothing, subsampling=subsampling, 
                          scale=scale, color=color, nTraces=nTraces, 
                          scale_side=scale_side, name=name)


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
    """ LFP in depth bands (see add_electrical_series) """
    add_electrical_series(data, 'LFP', tlim, ax,
                          fig_fraction_start=fig_fraction_start, 
                          fig_fraction=fig_fraction, 
                          vicinity_factor=vicinity_factor, 
                          smoothing=smoothing, subsampling=subsampling, 
                          scale=scale, color=color, nTraces=nTraces, 
                          scale_side=scale_side, name=name)
