# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt

from physion.dataviz.dataframe import tools as dv_tools
import physion.utils.plot_tools as pt

def add_Photodiode(dataframe, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., 
                   subsampling=10, 
                   color='#808080', 
                   name='photodiode'):

    dv_tools.plot_scaled_signal(dataframe, ax, 'Photodiode-Signal', tlim, 1,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_label='',
                                color=color)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Electrophy(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1.,
                   color='k',
                   scale_side='left',
                   name='LFP'):

    pass



def add_Locomotion(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2,
                   scale_side='left', 
                   scale_label='',
                   color='#1f77b4', name='run. speed'):

    dv_tools.plot_scaled_signal(data, ax, 'Running-Speed', tlim, 
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                scale_label=scale_label,
                                color=color)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)
   

def add_Pupil(data, tlim, ax,
              fig_fraction_start=0., fig_fraction=1., subsampling=2,
              scale_side='left',
              scale_label='',
              color='red', name='pupil diam.'):

    dv_tools.plot_scaled_signal(data, ax, 'Pupil-diameter', tlim, 
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                scale_label=scale_label,
                                color=color)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)
   
def add_GazeMovement(data, tlim, ax,
                     fig_fraction_start=0., fig_fraction=1.,
                     scale_side='left',
                     scale_label='',
                     color='#ff7f0e', name='gaze mov.'):

    dv_tools.plot_scaled_signal(data, ax, 'Gaze-Position', tlim, 
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                scale_label=scale_label,
                                color=color)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_FaceMotion(data, tlim, ax,
                   scale_side='left',
                   scale_label='',
                   fig_fraction_start=0., fig_fraction=1.,
                   color='#9467bd', name='facemotion'):

    dv_tools.plot_scaled_signal(data, ax, 'Whisking', tlim, 
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                scale_label=scale_label,
                                color=color)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_VisualStim(dataframe, tlim, ax,
                   fig_fraction_start=0., fig_fraction=0.05,
                   color='k',
                   name='visual stim.'):

    tcond = (dataframe['time']>=tlim[0]) & (dataframe['time']<=tlim[1])
    t = dataframe['time'][tcond]
    y = 1.*dataframe['visualStimFlag'][tcond]

    ax.fill_between(t, 0*y, y, lw=0, color='k', alpha=0.1)
    
    
def add_CaImagingRaster(dataframe, tlim, ax,
                        fig_fraction_start=0., fig_fraction=1., 
                        color='green',
                        subquantity='Fluorescence', 
                        roiIndices='all', subquantity_args={},
                        cmap=plt.cm.PiYG, zlim=[-1, 2],
                        axb=None,
                        bar_inset_start=-0.08, bar_inset_width=0.01,
                        normalization='None', subsampling=1,
                        name=''):

    raster = np.array([dataframe['dFoF-ROI%i'%i] for i in range(dataframe.vNrois)])

    indices = np.flatnonzero((dataframe['time']>tlim[0]) & (dataframe['time']<tlim[1]))
    
    ims = ax.imshow(raster[:,indices], origin='lower', cmap=cmap,
              aspect='auto', interpolation='none', 
              vmin=zlim[0], vmax=zlim[1],
              extent=(tlim[0], tlim[1],
                      fig_fraction_start, fig_fraction_start+fig_fraction))

    if axb is None:
        axb = pt.inset(ax, [bar_inset_start, fig_fraction_start+.2*fig_fraction,
                            bar_inset_width, .6*fig_fraction], facecolor='w')

    cb = plt.colorbar(ims, cax=axb)
    axb.set_ylabel('Zscore $\Delta$F/F', fontsize=8)

    dv_tools.add_name_annotation(ax, name, tlim,
            fig_fraction, fig_fraction_start, rotation=90)

    ax.annotate('1', (tlim[1], fig_fraction_start), xycoords='data')
    ax.annotate('%i' % raster.shape[0],
                (tlim[1], fig_fraction_start+fig_fraction), va='top', xycoords='data')
    ax.annotate('rois', 
                (tlim[1], fig_fraction_start+fig_fraction/2.),
                va='center',
                # rotation=-90,
                xycoords='data',
                fontsize=8)

def add_CaImaging(dataframe, tlim, ax,
                  fig_fraction_start=0., fig_fraction=1., color='green',
                  roiIndices='all', 
                  scale_side='left',
                  scale_label='',
                  vicinity_factor=1, subsampling=1, name='[Ca] imaging',
                  annotation_side='right'):

    if (type(roiIndices)==str) and roiIndices=='all':
        roiIndices = np.arange(dataframe.vNrois)
        
    if color=='tab':
        COLORS = [plt.cm.tab10(n%10) for n in range(len(roiIndices))]
    else:
        COLORS = [str(color) for n in range(len(roiIndices))]

    indices = np.flatnonzero((dataframe['time']>tlim[0]) & (dataframe['time']<tlim[1]))
    t = dataframe['time'][indices]

    for n, ir in zip(range(len(roiIndices))[::-1], roiIndices[::-1]):

        ypos = n*fig_fraction/len(roiIndices)/vicinity_factor+\
                fig_fraction_start # bottom position

        dv_tools.plot_scaled_signal(dataframe, ax, 'dFoF-ROI%i'%ir, tlim,
                              ax_fraction_extent=fig_fraction/len(roiIndices),
                              ax_fraction_start=ypos,
                              color=color, 
                              scale_side=scale_side,
                              scale_label=scale_label)

        dv_tools.add_name_annotation(ax, 'roi #%i'%(ir+1), 
                                     tlim, fig_fraction/len(roiIndices), ypos,
                                     color=color, side=annotation_side)
        
    

def find_default_plot_settings(data, Nmax=7):
    settings = {}

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
    
    show = {}
    for key in settings:
        settings[key]['fig_fraction_start'] = fstart
        settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
        fstart += settings[key]['fig_fraction']
        
    for key in settings:
        if not 'no-show' in key:
            exec('add_%s(data, tlim, ax, **settings[key])' % key)

    # time scale bar
    if Tbar==0.:
        Tbar = np.max([int((tlim[1]-tlim[0])/30.), 1])

    ax.plot([dv_tools.shifted_start(tlim), dv_tools.shifted_start(tlim)+Tbar], [1.,1.], lw=1, color='k')
    ax.annotate((' %is' % Tbar if Tbar>=1 else  '%.1fs' % Tbar) ,
                [dv_tools.shifted_start(tlim), 1.02], color='k')#, fontsize=9)
    
    ax.axis('off')
    ax.set_xlim([dv_tools.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
    ax.set_ylim([-0.05,1.05])

    if zoom_area is not None:
        ax.fill_between(zoom_area, [0,0], [1,1],  color='k', alpha=.2, lw=0)
    
    return fig, ax


    

    
     
