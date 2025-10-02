# general modules
# import os, sys, pathlib
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import filters

from physion.dataviz import tools as dv_tools
import physion.utils.plot_tools as pt

def add_CaImagingRaster(data, tlim, ax, raster=None,
                        #
                        fig_fraction_start=0., 
                        fig_fraction=1., 
                        color='green',
                        # 
                        subquantity='Fluorescence', 
                        roiIndices='all', 
                        subquantity_args={},
                        #
                        cmap=plt.cm.binary, 
                        axb=None,
                        bar_inset_start=-0.01, bar_inset_width=0.01,
                        normalization='None', subsampling=1,
                        name=''):

    if (subquantity in ['Fluorescence', 'rawFluo']) and (raster is None):
        if (type(roiIndices)==str) and (roiIndices=='all'):
            raster = data.rawFluo[:,:]
        else:
            raster = data.rawFluo[roiIndices,:]
            
    elif (subquantity in ['Neuropil', 'neuropil']) and (raster is None):
        if (type(roiIndices)==str) and (roiIndices=='all'):
            raster = data.neuropil[:,:]
        else:
            raster = data.neuropil[roiIndices,:]

    elif (subquantity in ['dFoF', 'dF/F']) and (raster is None):
        if not hasattr(data, 'dFoF'):
            data.build_dFoF(**subquantity_args)
        if (type(roiIndices)==str) and (roiIndices=='all'):
            raster = data.dFoF[:,:]
        else:
            raster = data.dFoF[roiIndices,:]
            
        roiIndices = np.arange(data.nROIs)

    elif (type(roiIndices)==str) and (roiIndices=='all') and (subquantity in ['dFoF', 'dF/F']):
        roiIndices = np.arange(data.nROIs)
        
    if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:
        raster = np.array([(raster[i,:]-np.min(raster[i,:]))/(np.max(raster[i,:])-\
                                np.min(raster[i,:])) for i in range(raster.shape[0])])
        
    indices=np.arange(*dv_tools.convert_times_to_indices(*tlim,
                                data.Neuropil, axis=1))[::subsampling]
    
    ims = ax.imshow(raster[:,indices], origin='lower', cmap=cmap,
              aspect='auto', interpolation='none', vmin=0, vmax=1,
              extent=(dv_tools.convert_index_to_time(indices[0], data.Neuropil),
                      dv_tools.convert_index_to_time(indices[-1], data.Neuropil),
                      fig_fraction_start, fig_fraction_start+fig_fraction))

    if normalization in ['per line', 'per-line', 'per cell', 'per-cell']:

        if axb is None:
            axb = pt.inset(ax, [bar_inset_start, fig_fraction_start+.2*fig_fraction,
                                bar_inset_width, .6*fig_fraction], facecolor='w')

        cb = plt.colorbar(ims, cax=axb)
        cb.set_ticks([])
        axb.set_ylabel('$\\Delta$F/F' if (subquantity in ['dFoF', 'dF/F']) else ' fluo.', fontsize=9)
        axb.annotate('max', (0.5,1.1), fontsize=7,
                xycoords='axes fraction', ha='center')
        axb.annotate('min', (0.5,-0.1), fontsize=7, va='top',
                xycoords='axes fraction', ha='center')

        
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, rotation=90)

    ax.annotate('1', (tlim[1], fig_fraction_start), xycoords='data')
    ax.annotate('%i' % raster.shape[0],
                (tlim[1], fig_fraction_start+fig_fraction), va='top', xycoords='data')
    ax.annotate('ROIs', 
                (tlim[1], fig_fraction_start+fig_fraction/2.),
                va='center',
                rotation=-90,
                xycoords='data',
                fontsize=8)

    
    
def add_CaImaging(data, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., color='green',
                  subquantity='Fluorescence', 
                  roiIndices='all', 
                  dFoF_args={},
                  scale_side='left',
                  vicinity_factor=1, 
                  subsampling=1, 
                  name='[Ca] imaging',
                  annotation_side='left',
                  sigma=0):

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

        ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+\
                fig_fraction_start # bottom position

        if (subquantity in ['dF/F', 'dFoF']):
            if sigma > 0 :
                y = filters.gaussian_filter1d(data.dFoF[ir, np.arange(i1,i2)][::subsampling], sigma)
            else : 
                y = data.dFoF[ir, np.arange(i1,i2)][::subsampling]
            dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 1.,
                              ax_fraction_extent=fig_fraction/len(roiIndices),
                              ax_fraction_start=ypos,
                              color=color, 
                              scale_side=scale_side,
                             scale_unit_string=('%.0f$\\Delta$F/F' if (n==0) else ' '))
        else:
            if sigma > 0 :
                y = filters.gaussian_filter1d(data.rawFluo[ir,np.arange(i1,i2)][::subsampling], sigma)
            else : 
                y = data.rawFluo[ir,np.arange(i1,i2)][::subsampling]
            dv_tools.plot_scaled_signal(data, ax, t, y, tlim, 1.,
                   ax_fraction_extent=fig_fraction/len(roiIndices),
                   ax_fraction_start=ypos, color=color,
                   scale_side=scale_side,
                   scale_unit_string=('fluo (a.u.)' if (n==0) else ''))

        if annotation_side!='':
            dv_tools.add_name_annotation(data, ax, 
                    'roi #%i'%(ir+1), tlim, fig_fraction/len(roiIndices),
                                         ypos, color=color, 
                                         side=annotation_side)
        
        

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
        y = data.rawFluo.sum(axis=0)[np.arange(i1,i2)][::subsampling]

    dv_tools.plot_scaled_signal(data, ax, t, y, tlim, 1., fig_fraction, fig_fraction_start, color=color,
                            scale_unit_string=('%.0fdF/F' if subquantity in ['dF/F', 'dFoF'] else ''))
    dv_tools.add_name_annotation(data, ax, name, tlim, fig_fraction, fig_fraction_start, color=color)

###-------------------------------------
### ----- IMAGING PLOT components -----
###-------------------------------------

def find_full_roi_coords(data, roiIndex):

    indices = np.arange(\
        (data.pixel_masks_index[roiIndex-1] if roiIndex>0 else 0),
        (data.pixel_masks_index[roiIndex] if\
                roiIndex<data.original_nROIs else len(data.pixel_masks_index)))
    return [data.pixel_masks[ii][1] for ii in indices],\
                [data.pixel_masks[ii][0] for ii in indices]

def find_roi_coords(data, roiIndex):
    x, y = find_full_roi_coords(data, roiIndex)
    return np.mean(y), np.mean(x), np.std(y), np.std(x)

def find_roi_extent(data, roiIndex, 
                    force_square=False,
                    roi_zoom_factor=10.):

    mx, my, sx, sy = find_roi_coords(data, roiIndex)

    if force_square:
        sx = np.mean([sx, sy])
        sy = sx

    return np.array((mx-roi_zoom_factor*sx, mx+roi_zoom_factor*sx,
                     my-roi_zoom_factor*sy, my+roi_zoom_factor*sy), dtype=int)


def find_roi_cond(data, roiIndex, 
                  force_square=False,
                  roi_zoom_factor=10.):

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
    ellipse = plt.Circle((my, mx), 
                         size_factor*(sy+sx), 
                         edgecolor='lightgray', 
                         facecolor='none', lw=roi_lw)
    ax.add_patch(ellipse)

def get_FOV_image(data, key):

    img = getattr(\
            getattr(data.nwbfile.processing['ophys'],
                  'data_interfaces')['Backgrounds_0'],
                  'images')[key][:]

    extent=(0, img.shape[1], 
            0, img.shape[0])
    
    return img, extent


def show_CaImaging_FOV(data, 
                       key='meanImg', 
                       NL=1, 
                       cmap=pt.get_linear_colormap('k', 'g'), 
                       ax=None,
                       roiIndex=None, 
                       with_ROI_annotation=False,
                       with_annotation=True,
                       fig_args=dict(ax_scale=(1.4,2.4)),
                       ):

    # set up axes 
    if ax is None:
        fig, ax = pt.figure(**fig_args)
    else:
        fig = None
    ax.axis('equal')

    img, extent = get_FOV_image(data, key)

    img = (img-img.min())/(img.max()-img.min())
    img = np.power(img, 1/NL)
    img = ax.imshow(img, vmin=0, vmax=1, 
                    cmap=cmap, 
                    aspect='equal', 
                    interpolation='none', 
                    origin='lower', 
                    extent=extent)
    ax.axis('off')

    if roiIndex is not None:

        if type(roiIndex)==int:
            roiIndex = [roiIndex]

        for roi in roiIndex:
            x, y = find_full_roi_coords(data, roi)
            ax.plot(x, y, '.', 
                    # color=plt.cm.tab10(roiIndex%10), 
                    # color=plt.cm.hsv(np.random.uniform(0,1)),
                    color=plt.cm.autumn(np.random.uniform(0,1)),
                    alpha=0.5,
                    ms=0.1)

            if with_ROI_annotation:
                ax.annotate('%i' % (roi+1), (np.mean(x), np.mean(y)), 
                            color='w', fontsize=7)

    if with_annotation:
        ax.annotate('%i ROIs' % data.nROIs, 
                    (0, 0), xycoords='axes fraction', 
                    rotation=90, ha='right')
        ax.set_title(key)
    
    return fig, ax, img


def show_singleROI_in_FOV(data, 
                       key='meanImg', 
                       NL=1, 
                       cmap=pt.get_linear_colormap('k', 'g'), 
                       ax=None,
                       roiIndex=None, 
                       roi_zoom_factor=10,
                       roi_lw=3,
                       with_ROI_annotation=False,
                       with_annotation=True,
                       with_roi_zoom=False,
                       fig_args=dict(ax_scale=(1.4,2.4)),
                       ):

    # set up axes 
    if ax is None:
        fig, ax = pt.figure(**fig_args)
    else:
        fig = None
    ax.axis('equal')

    # get FOV image
    img = getattr(\
            getattr(data.nwbfile.processing['ophys'],
                  'data_interfaces')['Backgrounds_0'],
                  'images')[key][:]

    extent=(0, img.shape[1], 
            0, img.shape[0])

    if with_roi_zoom and type(roiIndex)==int:
        zoom_cond, zoom_cond_shape = find_roi_cond(data, roiIndex,
                                            roi_zoom_factor=roi_zoom_factor)
        img = img[zoom_cond].reshape(*zoom_cond_shape)
        extent=find_roi_extent(data, roiIndex,
                               roi_zoom_factor=roi_zoom_factor)
    
    img = (img-img.min())/(img.max()-img.min())
    img = np.power(img, 1/NL)
    img = ax.imshow(img, vmin=0, vmax=1, 
                    cmap=cmap, 
                    aspect='equal', 
                    interpolation='none', 
                    origin='lower', 
                    extent=extent)
    ax.axis('off')

    if type(roiIndex)==int:
        # if single ROI
        add_roi_ellipse(data, roiIndex, 
                        ax, roi_lw=roi_lw)

    else:
        # multiple ROIs
        if roiIndex is None:
            roiIndex = np.arange(data.nROIs)

        for roi in roiIndex:
            x, y = find_full_roi_coords(data, roi)
            ax.plot(x, y, '.', 
                    # color=plt.cm.tab10(roiIndex%10), 
                    # color=plt.cm.hsv(np.random.uniform(0,1)),
                    color=plt.cm.autumn(np.random.uniform(0,1)),
                    alpha=0.5,
                    ms=0.1)

            if with_ROI_annotation:
                ax.annotate('%i' % (roi+1), (np.mean(x), np.mean(y)), 
                            color='w', fontsize=7)

    if with_annotation:
        ax.annotate('%i ROIs' % data.nROIs, 
                    (0, 0), xycoords='axes fraction', 
                    rotation=90, ha='right')
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
    parser.add_argument('-o', "--ops", 
                        default='raw', help='')
    parser.add_argument("--tlim", type=float, nargs='*', 
                        default=[10, 50], help='')
    parser.add_argument('-e', "--episode", type=int, default=0)
    parser.add_argument('-nmax', "--Nmax", type=int, default=20)
    parser.add_argument("--Npanels", type=int, default=8)
    parser.add_argument('-roi', "--roiIndex", type=int, default=0)
    parser.add_argument('-pid', "--protocol_id", 
                        type=int, default=0)
    parser.add_argument('-q', "--quantity", 
                        type=str, default='dFoF')
    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity", 
                        action="store_true")

    args = parser.parse_args()
    
    import physion

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

        data = physion.analysis.read_NWB.Data(args.datafile)
        fig, AX = data.show_VisualStim(args.tlim, 
                                       Npanels=args.Npanels)
        fig2 = data.visual_stim.plot_stim_picture(args.episode)
        print('interval [%.1f, %.1f] ' % (\
         data.nwbfile.stimulus['time_start_realigned'].data[args.episode],
         data.nwbfile.stimulus['time_stop_realigned'].data[args.episode]))
        
    elif args.ops=='FOV':

        data = physion.analysis.read_NWB.Data(args.datafile)
        fig, ax = plt.subplots(1, figsize=(5,4))
        show_CaImaging_FOV(data, 'meanImg', NL=3,
                           roiIndices='all',
                           with_ROI_annotation=True,
                           ax=ax)

    else:
        print(' option not recognized !')
       
    plt.show()
