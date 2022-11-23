# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
plt.style.use(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'utils', 'matplotlib_style.py'))

from physion.dataviz import tools as dv_tools

from physion.analysis import read_NWB#, tools, stat_tools
from physion.visual_stim.build import build_stim


def shifted_start(tlim, frac_shift=0.01):
    return tlim[0]-frac_shift*(tlim[1]-tlim[0])


def plot_scaled_signal(data, 
                       ax, t, signal,
                       tlim, scale_bar,
                       ax_fraction_extent, ax_fraction_start,
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

    ax.plot(t, ax_fraction_start+(signal-min_signal)*ax_fraction_extent/scale_range, color=color, lw=1)

    if scale_unit_string!='':
        ax.plot(shifted_start(tlim)*np.ones(2), ax_fraction_start+scale_bar*np.arange(2)*ax_fraction_extent/scale_range, color=color, lw=1)
    if '%' in scale_unit_string:
        ax.annotate(str(scale_unit_string+' ') % scale_bar,
                (shifted_start(tlim), ax_fraction_start),
                ha='right', color=color, va='center', xycoords='data')
    elif scale_unit_string!='':
        ax.annotate(scale_unit_string,
                (shifted_start(tlim), ax_fraction_start),
                ha='right', color=color, va='center', xycoords='data')


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


def add_Photodiode(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., 
                   subsampling=10, 
                   color='#808080', 
                   name='photodiode'):
    i1, i2 = dv_tools.convert_times_to_indices(*tlim, data.nwbfile.acquisition['Photodiode-Signal'])
    t = dv_tools.convert_index_to_time(range(i1,i2),
            data.nwbfile.acquisition['Photodiode-Signal'])[::subsampling]
    y = data.nwbfile.acquisition['Photodiode-Signal'].data[i1:i2][::subsampling]
    
    plot_scaled_signal(data,ax, t, y, tlim, 1e-5,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='')        
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Electrophy(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2, color='k',
                   name='LFP'):
    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.acquisition['Electrophysiological-Signal'])
    t = dv_tools.convert_index_to_time(range(i1,i2),
            data.nwbfile.acquisition['Electrophysiological-Signal'])[::subsampling]
    y = data.nwbfile.acquisition['Electrophysiological-Signal'].data[i1:i2][::subsampling]

    plot_scaled_signal(data,ax, t, y, tlim, 0.2,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmV')        
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Locomotion(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2,
                   speed_scale_bar=1, # cm/s
                   color='#1f77b4', name='run. speed'):

    if not hasattr(data, 'running_speed'):
        data.build_running_speed()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.acquisition['Running-Speed'])
    x, y = data.t_running_speed[i1:i2][::subsampling], data.running_speed[i1:i2][::subsampling]

    plot_scaled_signal(data,ax, x, y, tlim, speed_scale_bar,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fcm/s')        
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)
   

def add_Pupil(data, tlim, ax,
              fig_fraction_start=0., fig_fraction=1., subsampling=2,
              pupil_scale_bar = 0.5, # scale bar in mm
              color='red', name='pupil diam.'):

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['Pupil'].data_interfaces['cx'])

    if not hasattr(data, 'pupil_diameter'):
        data.build_pupil_diameter()

    x, y = data.t_pupil[i1:i2][::subsampling], data.pupil_diameter[i1:i2][::subsampling]

    plot_scaled_signal(data,ax, x, y, tlim, pupil_scale_bar,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')        
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_GazeMovement(data, tlim, ax,
                     fig_fraction_start=0., fig_fraction=1., subsampling=2,
                     gaze_scale_bar = 0.2, # scale bar in mm
                     color='#ff7f0e', name='gaze mov.'):

    if not hasattr(data, 'gaze_movement'):
        data.build_gaze_movement()
    
    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['Pupil'].data_interfaces['cx'])

    x, y = data.t_pupil[i1:i2][::subsampling], data.gaze_movement[i1:i2][::subsampling]

    plot_scaled_signal(data,ax, x, y, tlim, gaze_scale_bar,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='%.1fmm')
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_FaceMotion(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2, color='#9467bd', name='facemotion'):

    if not hasattr(data, 'facemotion'):
        data.build_facemotion()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])

    x, y = data.t_facemotion[i1:i2][::subsampling], data.facemotion[i1:i2][::subsampling]

    plot_scaled_signal(data, ax, x, y, tlim, 1.,
            fig_fraction, fig_fraction_start, color=color, scale_unit_string='') # no scale bar here
    
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)

    
def add_CaImagingRaster(data, tlim, ax, raster=None,
                        fig_fraction_start=0., fig_fraction=1., color='green',
                        subquantity='Fluorescence', roiIndices='all', subquantity_args={},
                        cmap=plt.cm.binary,
                        normalization='None', subsampling=1,
                        name='\nROIs'):

    if subquantity=='Fluorescence' and (raster is None):
        if (roiIndices=='all'):
            raster = data.Fluorescence.data[:,:]
        else:
            raster = data.Fluorescence.data[roiIndices,:]
            
    elif (subquantity in ['dFoF', 'dF/F']) and (raster is None):
        if not hasattr(data, 'dFoF'):
            data.build_dFoF(**subquantity_args)
        if (roiIndices=='all'):
            raster = data.dFoF[:,:]
        else:
            raster = data.dFoF[roiIndices,:]
            
        roiIndices = np.arange(data.iscell.sum())

    elif (roiIndices=='all') and (subquantity in ['dFoF', 'dF/F']):
        roiIndices = np.arange(data.nROIs)
        
    if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
        raster = np.array([(raster[i,:]-np.min(raster[i,:]))/(np.max(raster[i,:])-\
                                np.min(raster[i,:])) for i in range(raster.shape[0])])
        
    indices=np.arange(*dv_tools.convert_times_to_indices(*tlim,
                                data.Neuropil, axis=1))[::subsampling]
    
    ax.imshow(raster[:,indices], origin='lower', cmap=cmap,
              aspect='auto', interpolation='none', vmin=0, vmax=1,
              extent=(dv_tools.convert_index_to_time(indices[0], data.Neuropil),
                      dv_tools.convert_index_to_time(indices[-1], data.Neuropil),
                      fig_fraction_start, fig_fraction_start+fig_fraction))

    # if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
        # _, axb = ge.bar_legend(ax,
                      # # X=[0,1], bounds=[0,1],
                      # continuous=False, colormap=cmap,
                      # colorbar_inset=dict(rect=[-.06,
                                       # fig_fraction_start+.2*fig_fraction,
                                       # .01,
                                       # .6*fig_fraction], facecolor=None),
                      # color_discretization=100, no_ticks=True, labelpad=4.,
                      # label=('$\Delta$F/F' if (subquantity in ['dFoF', 'dF/F']) else ' fluo.'),
                      # fontsize='small')
        # ge.annotate(axb, ' max', (1,1), size='x-small')
        # ge.annotate(axb, ' min', (1,0), size='x-small', va='top')
        
    add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, rotation=90)

    ax.annotate('1', (tlim[1], fig_fraction_start), xycoords='data')
    ax.annotate('%i' % raster.shape[0],
                (tlim[1], fig_fraction_start+fig_fraction), va='top', xycoords='data')
    
    
def add_CaImaging(data, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., color='green',
                  subquantity='Fluorescence', roiIndices='all', dFoF_args={},
                  vicinity_factor=1, subsampling=1, name='[Ca] imaging',
                  annotation_side='right'):

    if (subquantity in ['dF/F', 'dFoF']) and (not hasattr(data, 'dFoF')):
        data.build_dFoF(**dFoF_args)
        
    if (type(roiIndices)==str) and roiIndices=='all':
        roiIndices = data.valid_roiIndices
        
    if color=='tab':
        COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
    else:
        COLORS = [str(color) for n in range(len(roiIndices))]

    i1, i2 = dv_tools.convert_times_to_indices(*tlim, data.Neuropil, axis=1)
    t = np.array(data.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]

    for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):

        ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+fig_fraction_start # bottom position

        if (subquantity in ['dF/F', 'dFoF']):
            y = data.dFoF[ir, np.arange(i1,i2)][::subsampling]
            plot_scaled_signal(data,ax, t, y, tlim, 1., fig_fraction/len(roiIndices), ypos, color=color,
                                    scale_unit_string=('%.0f$\Delta$F/F' if (n==0) else ' '))
        else:
            y = data.Fluorescence.data[ir, np.arange(i1,i2)][::subsampling]
            plot_scaled_signal(data, ax, t, y, tlim, 1., fig_fraction/len(roiIndices), ypos, color=color,
                                    scale_unit_string=('fluo (a.u.)' if (n==0) else ''))

        add_name_annotation(data, ax, 'ROI#%i'%(ir+1), tlim, fig_fraction/len(roiIndices), ypos,
                color=color, side=annotation_side)
        
    # ax.annotate(name, (shifted_start(tlim), fig_fraction/2.+fig_fraction_start), color=color,
    #             xycoords='data', ha='right', va='center', rotation=90)
        

def add_CaImagingSum(data, tlim, ax,
                     fig_fraction_start=0., fig_fraction=1., color='green',
                     subquantity='Fluorescence', subsampling=1,
                     name='Sum [Ca]'):
    
    if (subquantity in ['dF/F', 'dFoF']) and (not hasattr(data, 'dFoF')):
        data.build_dFoF()
        
    i1, i2 = dv_tools.convert_times_to_indices(*tlim, data.Neuropil, axis=1)
    t = np.array(data.Neuropil.timestamps[:])[np.arange(i1,i2)][::subsampling]
    
    if (subquantity in ['dF/F', 'dFoF']):
        y = data.dFoF.sum(axis=0)[np.arange(i1,i2)][::subsampling]
    else:
        y = data.Fluorescence.data[:,:].sum(axis=0)[np.arange(i1,i2)][::subsampling]

    plot_scaled_signal(data, ax, t, y, tlim, 1., fig_fraction, fig_fraction_start, color=color,
                            scale_unit_string=('%.0fdF/F' if subquantity in ['dF/F', 'dFoF'] else ''))
    add_name_annotation(data, ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

    
def add_VisualStim(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=0.05, size=0.1,
                   with_screen_inset=True,
                   color='k', name='visual stim.'):
    if data.visual_stim is None:
        data.init_visual_stim()
    # cond = (data.nwbfile.stimulus['time_start_realigned'].data[:]>tlim[0]) &\
        # (data.nwbfile.stimulus['time_stop_realigned'].data[:]<tlim[1])
    cond = (data.nwbfile.stimulus['time_start_realigned'].data[:]<tlim[1]) &\
        (data.nwbfile.stimulus['time_stop_realigned'].data[:]>tlim[0])
    ylevel = fig_fraction_start+fig_fraction/2.
    sx, sy = data.visual_stim.screen['resolution']
    ax_pos = ax.get_position()
    for i in np.arange(data.nwbfile.stimulus['time_start_realigned'].num_samples)[cond]:
        tstart = data.nwbfile.stimulus['time_start_realigned'].data[i]
        tstop = data.nwbfile.stimulus['time_stop_realigned'].data[i]
        # ax.plot([tstart, tstop], [ylevel, ylevel], color=color)
        ax.fill_between([tstart, tstop], [0,0], np.zeros(2)+ylevel,
                        lw=0, alpha=0.05, color=color)
        if with_screen_inset:
            axi = ax.inset_axes([tstart, 1.01, (tstop-tstart), size], transform=ax.transData)
            axi.axis('equal')
            data.visual_stim.plot_stim_picture(i, ax=axi)
    ax.annotate(' '+name, (tlim[1], fig_fraction+fig_fraction_start), color=color, xycoords='data')

    
def show_VisualStim(data, tlim,
                    Npanels=8):
    
    if data.visual_stim is None:
        data.init_visual_stim()

    fig, AX = plt.subplots(Npanels,1)

    label={'degree':20,
           'shift_factor':0.03,
           'lw':0.5, 'fontsize':7}
    
    for i, ti in enumerate(np.linspace(*tlim, Npanels)):
        iEp = data.find_episode_from_time(ti)
        tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
        if iEp>=0:
            data.visual_stim.show_frame(iEp, ax=AX[i],
                                        time_from_episode_start=ti-tEp,
                                        label=label)
        AX[i].set_title('%.1fs' % ti, fontsize=6)
        AX[i].axis('off')
        label=None
        
    return fig, AX


def find_default_plot_settings(data, Nmax=7):
    settings = {}

    if data.metadata['VisualStim']:
        settings['Photodiode'] = dict(fig_fraction=.5, subsampling=1, color='grey')

    if data.metadata['Locomotion']:
        settings['Locomotion'] = dict(fig_fraction=1, subsampling=1, color='#1f77b4')

    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion'] = dict(fig_fraction=1, subsampling=10, color='purple')

    if 'Pupil' in data.nwbfile.processing:
        settings['GazeMovement'] = dict(fig_fraction=0.5, subsampling=1, color='#ff7f0e')

    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil']= dict(fig_fraction=2, subsampling=1, color='#d62728')

    if 'ophys' in data.nwbfile.processing:
        settings['CaImaging'] = dict(fig_fraction=4, subsampling=1, 
                                     subquantity='dF/F', color='#2ca02c',
                                     roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)),
                                         np.min([Nmax, data.iscell.sum()]), replace=False)))

    if 'ophys' in data.nwbfile.processing:
        settings['CaImagingRaster'] = dict(fig_fraction=3, subsampling=1,
                                           roiIndices='all',
                                           normalization='per-line',
                                           subquantity='dF/F')

    if data.metadata['VisualStim']:
        settings['VisualStim'] = dict(fig_fraction=.5, color='black')

    return settings 

def plot(data, 
         tlim=[0,100],
         settings = {},
         figsize=(3,5), Tbar=0., zoom_area=None,
         ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10,4))
    else:
        fig = None
        
    fig_fraction_full, fstart = np.sum([settings[key]['fig_fraction'] for key in settings]), 0
    
    for key in settings:
        settings[key]['fig_fraction_start'] = fstart
        settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
        fstart += settings[key]['fig_fraction']
        
    for key in settings:
        exec('add_%s(data, tlim, ax, **settings[key])' % key)

    # time scale bar
    if Tbar==0.:
        Tbar = np.max([int((tlim[1]-tlim[0])/30.), 1])

    ax.plot([shifted_start(tlim), shifted_start(tlim)+Tbar], [1.,1.], lw=1, color='k')
    ax.annotate((' %is' % Tbar if Tbar>=1 else  '%.1fs' % Tbar) ,
                [shifted_start(tlim), 1.02], color='k', fontsize=9)
    
    ax.axis('off')
    ax.set_xlim([shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
    ax.set_ylim([-0.05,1.05])

    if zoom_area is not None:
        ax.fill_between(zoom_area, [0,0], [1,1],  color='k', alpha=.2, lw=0)
    
    return fig, ax


###-------------------------------------
### ----- IMAGING PLOT components -----
###-------------------------------------

def find_full_roi_coords(data, roiIndex):

    indices = np.arange((data.pixel_masks_index[roiIndex-1] if roiIndex>0 else 0),
                        (data.pixel_masks_index[roiIndex] if roiIndex<len(data.valid_roiIndices) else len(data.pixel_masks_index)))
    return [data.pixel_masks[ii][1] for ii in indices],  [data.pixel_masks[ii][0] for ii in indices]

def find_roi_coords(data, roiIndex):
    x, y = data.find_full_roi_coords(roiIndex)
    return np.mean(y), np.mean(x), np.std(y), np.std(x)

def find_roi_extent(data, roiIndex, roi_zoom_factor=10.):

    mx, my, sx, sy = find_roi_coords(data, roiIndex)

    return np.array((mx-roi_zoom_factor*sx, mx+roi_zoom_factor*sx,
                     my-roi_zoom_factor*sy, my+roi_zoom_factor*sy), dtype=int)


def find_roi_cond(data, roiIndex, roi_zoom_factor=10.):

    mx, my, sx, sy = find_roi_coords(data, roiIndex)

    img_shape = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'][:].shape

    x, y = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')
    cond = (x>=(mx-roi_zoom_factor*sx)) &\
            (x<=(mx+roi_zoom_factor*sx)) &\
           (y>=(my-roi_zoom_factor*sy)) &\
            (y<=(my+roi_zoom_factor*sy)) 
    roi_zoom_shape = (len(np.unique(x[cond])), len(np.unique(y[cond])))

    return cond, roi_zoom_shape

def add_roi_ellipse(data, roiIndex, ax,
                    size_factor=1.5,
                    roi_lw=3):

    mx, my, sx, sy = find_roi_coords(data, roiIndex)
    ellipse = plt.Circle((mx, my), size_factor*(sy+sx), edgecolor='lightgray', facecolor='none', lw=roi_lw)
    ax.add_patch(ellipse)

def show_CaImaging_FOV(data, key='meanImg', NL=1, cmap='viridis', ax=None,
                       roiIndex=None, roiIndices=[],
                       roi_zoom_factor=10,
                       roi_lw=3,
                       with_roi_zoom=False,):
    
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None
    ax.axis('equal')

    img = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images[key][:]
    extent=(0,img.shape[1], 0, img.shape[0])

    if with_roi_zoom and roiIndex is not None:
        zoom_cond, zoom_cond_shape = find_roi_cond(data, roiIndex,
                                            roi_zoom_factor=roi_zoom_factor)
        img = img[zoom_cond].reshape(*zoom_cond_shape)
        extent=find_roi_extent(data, roiIndex,
                               roi_zoom_factor=roi_zoom_factor)
    
    img = (img-img.min())/(img.max()-img.min())
    img = np.power(img, 1/NL)
    img = ax.imshow(img, vmin=0, vmax=1, cmap=cmap, aspect='equal', interpolation='none', 
            origin='lower',
            extent=extent)
    ax.axis('off')

    if roiIndex is not None:
        add_roi_ellipse(data, roiIndex, ax, roi_lw=roi_lw)

    if roiIndices=='all':
        roiIndices = data.valid_roiIndices

    for roiIndex in roiIndices:
        x, y = data.find_full_roi_coords(roiIndex)
        ax.plot(x, y, '.', 
                # color=plt.cm.tab10(roiIndex%10), 
                # color=plt.cm.hsv(np.random.uniform(0,1)),
                color=plt.cm.autumn(np.random.uniform(0,1)),
                alpha=0.5,
                ms=0.1)
    ax.annotate('%i ROIs' % np.sum(data.iscell), (0, 0), xycoords='axes fraction', rotation=90, ha='right')
    
    ax.set_title(key)
    
    return fig, ax, img


    
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

    
     
if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)
    parser.add_argument('-o', "--ops", default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', default=[10, 50], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=20)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-roi', "--roiIndex", type=int, default=0)
    parser.add_argument('-pid', "--protocol_id", type=int, default=0)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()
    
    if args.ops=='raw':

        data = MultimodalData(args.datafile)

        # data.plot_raw_data(args.tlim, 
                  # settings={'CaImagingRaster':dict(fig_fraction=4, subsampling=1,
                                                   # roiIndices='all',
                                                   # normalization='per-line',
                                                   # subquantity='dF/F'),
                            # 'CaImaging':dict(fig_fraction=3, subsampling=1, 
                                             # subquantity='dF/F', color='#2ca02c',
                                             # roiIndices=np.sort(np.random.choice(np.arange(np.sum(data.iscell)), np.min([args.Nmax, data.iscell.sum()]), replace=False))),
                            # 'Locomotion':dict(fig_fraction=1, subsampling=1, color='#1f77b4'),
                            # 'Pupil':dict(fig_fraction=2, subsampling=1, color='#d62728'),
                            # 'GazeMovement':dict(fig_fraction=1, subsampling=1, color='#ff7f0e'),
                            # 'Photodiode':dict(fig_fraction=.5, subsampling=1, color='grey'),
                            # 'VisualStim':dict(fig_fraction=.5, color='black')},
                            # Tbar=5)

        data.plot_raw_data(args.tlim)
        
    elif args.ops=='behavior':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=['running_speed', 'pupil_diameter'],
                                   prestim_duration=2,
                                   verbose=args.verbose)

        episodes.behavior_variability(episode_condition=episodes.find_episode_cond('Image-ID', 0),
                                      threshold2=0.1)


    elif args.ops=='trial-average':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity],
                                   prestim_duration=3,
                                   verbose=args.verbose)

        episodes.plot_trial_average(with_screen_inset=True,
                                    with_annotation=True,
                                    column_key='contrast')

        # episodes.plot_trial_average(column_key=['patch-radius', 'direction'],
                                    # row_key='patch-delay',
                                    # color_key='repeat',
                                    # roiIndex=52,
                                    # roiIndices=[52, 84, 85, 105, 115, 141, 149, 152, 155, 157],
                                    #     norm='MinMax-time-variations-after-trial-averaging-per-roi',
                                    #     with_std_over_rois=True, 
                                         # with_annotation=True,
                                         # with_stat_test=True,
                                         # verbose=args.verbose)

        # fig, AX = episodes.plot_trial_average(quantity=args.quantity,
                                              # roiIndex=args.roiIndex,
                                              # # roiIndices=[22,25,34,51,63],
                                              # # with_std_over_rois=True,
                                              # # norm='Zscore-time-variations-after-trial-averaging-per-roi',
                                              # column_key=list(episodes.varied_parameters.keys())[0],
                                              # xbar=1, xbarlabel='1s', 
                                              # ybar=1, ybarlabel='1 (Zscore, dF/F)',
                                              # with_stat_test=True,
                                              # with_annotation=True,
                                              # with_screen_inset=True,                                          
                                              # fig_preset='raw-traces-preset', color='#1f77b4', label='test\n')

    elif args.ops=='evoked-raster':

        episodes = EpisodeResponse(args.datafile,
                                   protocol_id=args.protocol_id,
                                   quantities=[args.quantity])

        VP = [key for key in episodes.varied_parameters if key!='repeat'] # varied parameters except rpeat

        # single stim
        episodes.plot_evoked_pattern(episodes.find_episode_cond(np.array(VP),
                                                                np.zeros(len(VP), dtype=int)),
                                     quantity=args.quantity)
        
        
    elif args.ops=='visual-stim':

        data = MultimodalData(args.datafile)
        fig, AX = data.show_VisualStim(args.tlim, Npanels=args.Npanels)
        fig2 = data.visual_stim.plot_stim_picture(args.episode)
        print('interval [%.1f, %.1f] ' % (data.nwbfile.stimulus['time_start_realigned'].data[args.episode],
                                          data.nwbfile.stimulus['time_stop_realigned'].data[args.episode]))
        
    elif args.ops=='FOV':

        data = MultimodalData(args.datafile)
        fig, ax = ge.figure(figsize=(2,4), left=0.1, bottom=0.1)
        data.show_CaImaging_FOV('meanImg', NL=3,
                cmap=ge.get_linear_colormap('k', 'lightgreen'), 
                roiIndices='all',
                ax=ax)
        ge.save_on_desktop(fig, 'fig.png', dpi=400)

    else:
        print(' option not recognized !')
        
    ge.show()




