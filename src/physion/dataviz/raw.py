# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt

from physion.dataviz import tools as dv_tools
from physion.dataviz.imaging import *

from physion.analysis import read_NWB

import matplotlib.patches as patches


def add_Photodiode(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1.,
                   subsampling=10,
                   color='#808080',
                   name='photodiode'):
    i1, i2 = dv_tools.convert_times_to_indices(*tlim, data.nwbfile.acquisition['Photodiode-Signal'])
    t = dv_tools.convert_index_to_time(range(i1,i2),
            data.nwbfile.acquisition['Photodiode-Signal'])[::subsampling]
    y = data.nwbfile.acquisition['Photodiode-Signal'].data[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 1e-5,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                color=color, scale_unit_string=' a.u.')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Electrophy(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2, color='k',
                   scale_side='left',
                   name='LFP'):
    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.acquisition['Electrophysiological-Signal'])
    t = dv_tools.convert_index_to_time(range(i1,i2),
            data.nwbfile.acquisition['Electrophysiological-Signal'])[::subsampling]
    y = data.nwbfile.acquisition['Electrophysiological-Signal'].data[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data,ax, t, y, tlim, 0.2,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
            color=color, scale_unit_string='%.1fmV')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Locomotion(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=1., subsampling=2,
                   speed_scale_bar=1, # cm/s
                   scale_side='left',
                   color='#1f77b4', name='run. speed'):

    if not hasattr(data, 'running_speed'):
        data.build_running_speed()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.acquisition['Running-Speed'])
    x, y = data.t_running_speed[i1:i2][::subsampling], data.running_speed[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data, ax, x, y,
                                tlim, speed_scale_bar,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                color=color, scale_unit_string='%icm/s')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_Pupil(data, tlim, ax,
              fig_fraction_start=0., fig_fraction=1., subsampling=2,
              pupil_scale_bar = 0.2, # scale bar in mm
              scale_side='left',
              color='red', name='pupil diam.'):

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['Pupil'].data_interfaces['cx'])

    if not hasattr(data, 'pupil_diameter'):
        data.build_pupil_diameter()

    x, y = data.t_pupil[i1:i2][::subsampling], data.pupil_diameter[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data,ax, x, y, tlim, pupil_scale_bar,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                color=color, scale_unit_string='%.1fmm')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_GazeMovement(data, tlim, ax,
                     fig_fraction_start=0., fig_fraction=1., subsampling=2,
                     gaze_scale_bar = 0.1, # scale bar in mm
                     scale_side='left',
                     color='#ff7f0e', name='gaze mov.'):

    if not hasattr(data, 'gaze_movement'):
        data.build_gaze_movement()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['Pupil'].data_interfaces['cx'])

    x, y = data.t_pupil[i1:i2][::subsampling], data.gaze_movement[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data,ax, x, y, tlim, gaze_scale_bar,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                color=color, scale_unit_string='%.1fmm')
    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_FaceMotion(data, tlim, ax,
                   scale_side='left',
                   fig_fraction_start=0., fig_fraction=1., subsampling=2, color='#9467bd', name='facemotion'):

    if not hasattr(data, 'facemotion'):
        data.build_facemotion()

    i1, i2 = dv_tools.convert_times_to_indices(*tlim,
            data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])

    x, y = data.t_facemotion[i1:i2][::subsampling], data.facemotion[i1:i2][::subsampling]

    dv_tools.plot_scaled_signal(data, ax, x, y, tlim, 1.,
                                ax_fraction_extent=fig_fraction,
                                ax_fraction_start=fig_fraction_start,
                                scale_side=scale_side,
                                color=color, scale_unit_string=' a.u.')

    dv_tools.add_name_annotation(data, ax, name, tlim,
            fig_fraction, fig_fraction_start, color=color)


def add_VisualStim(data, tlim, ax,
                   fig_fraction_start=0., fig_fraction=0.05, size=0.1,
                   with_screen_inset=True,
                   color='k', name='visual stim.'):

    if with_screen_inset and (data.visual_stim is None):
        data.init_visual_stim()

        sx, sy = data.visual_stim.screen['resolution']
        ax_pos = ax.get_position()

    # cond = (data.nwbfile.stimulus['time_start_realigned'].data[:]>tlim[0]) &\
        # (data.nwbfile.stimulus['time_stop_realigned'].data[:]<tlim[1])

    cond = (data.nwbfile.stimulus['time_start_realigned'].data[:,0]<tlim[1]) &\
        (data.nwbfile.stimulus['time_stop_realigned'].data[:,0]>tlim[0])

    ylevel = fig_fraction_start+fig_fraction/2.

    for i in np.arange(data.nwbfile.stimulus['time_start_realigned'].num_samples)[cond]:

        tstart = max([tlim[0], data.nwbfile.stimulus['time_start_realigned'].data[i,0]])
        tstop = min([tlim[1], data.nwbfile.stimulus['time_stop_realigned'].data[i,0]])
        # ax.plot([tstart, tstop], [ylevel, ylevel], color=color)
        ax.fill_between([tstart, tstop], [0,0], np.zeros(2)+ylevel,
                        lw=0, alpha=0.05, color=color)

        if with_screen_inset:
            axi = ax.inset_axes([tstart, 1.01, (tstop-tstart), size],\
                                transform=ax.transData)
            axi.axis('equal')
            data.visual_stim.plot_stim_picture(i, ax=axi)

    ax.annotate(' '+name, (tlim[1], fig_fraction+fig_fraction_start),\
                color=color, xycoords='data')


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
        tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp,0]
        if iEp>=0:
            data.visual_stim.show_frame(iEp, ax=AX[i],
                                        time_from_episode_start=ti-tEp,
                                        label=label)
        AX[i].set_title('%.1fs' % ti, fontsize=6)
        AX[i].axis('off')
        label=None

    return fig, AX


def find_default_plot_settings(data, 
                               with_subsampling=False,
                               Nmax=7):
    settings = {}

    if data.metadata['VisualStim']:
        settings['Photodiode'] = dict(fig_fraction=.5, 
                                      subsampling=100 if with_subsampling else 1, 
                                      color='grey')

    if data.metadata['Locomotion']:
        settings['Locomotion'] = dict(fig_fraction=1, 
                                      subsampling=10 if with_subsampling else 1, 
                                      color='#1f77b4')

    if 'FaceMotion' in data.nwbfile.processing:
        settings['FaceMotion'] = dict(fig_fraction=1, 
                                      subsampling=10 if with_subsampling else 1, 
                                      color='purple')

    if 'Pupil' in data.nwbfile.processing:
        settings['GazeMovement'] = dict(fig_fraction=0.5, 
                                        subsampling=10 if with_subsampling else 1, 
                                        color='#ff7f0e')

    if 'Pupil' in data.nwbfile.processing:
        settings['Pupil']= dict(fig_fraction=2, 
                                subsampling=10 if with_subsampling else 1, 
                                color='#d62728')

    if 'ophys' in data.nwbfile.processing:
        if not hasattr(data, 'dFoF'):
            data.build_dFoF()
        settings['CaImaging'] = dict(fig_fraction=4, 
                                     subsampling=10 if with_subsampling else 1, 
                                     subquantity='dFoF', color='#2ca02c',
                                     roiIndices=np.sort(np.random.choice(np.arange(data.nROIs),
                                          np.min([Nmax, data.nROIs]), replace=False)))

    if 'ophys' in data.nwbfile.processing:
        settings['CaImagingRaster'] = dict(fig_fraction=3, 
                                           subsampling=10 if with_subsampling else 1, 
                                           roiIndices='all',
                                           normalization='per-line',
                                           subquantity='dF/F')

    if data.metadata['VisualStim'] and not with_subsampling:
        settings['VisualStim'] = dict(fig_fraction=.5, 
                                      color='black')

    return settings

def plot(data,
         tlim=[0,100],
         settings = {},
         figsize=(9,6), 
         Tbar=0., zoom_area=None,
         ax=None, 
         grey=False, 
         black=False, 
         grey_co=[], 
         black_co=[]): #, 
         #state='both',
         #threshold = 0.5):
 
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    fig_fraction_full, fstart = np.sum([settings[key]['fig_fraction'] for key in settings]), 0

    for key in settings:
        settings[key]['fig_fraction_start'] = fstart
        settings[key]['fig_fraction'] = settings[key]['fig_fraction']/fig_fraction_full
        fstart += settings[key]['fig_fraction']

    for key in settings:
        exec('add_%s(data=data, tlim=tlim, ax=ax, **settings[key])' % key)
        #exec('add_%s(data=data, tlim=tlim, ax=ax, state=state, threshold=threshold, **settings[key])' % key)
    
    # time scale bar
    if Tbar==0.:
        Tbar = np.max([int((tlim[1]-tlim[0])/30.), 1])

    ax.plot([dv_tools.shifted_start(tlim), dv_tools.shifted_start(tlim)+Tbar], [1.,1.], lw=1, color='k')
    ax.annotate((' %is' % Tbar if Tbar>=1 else  '%.1fs' % Tbar) ,
                [dv_tools.shifted_start(tlim), 1.02], color='k')#, fontsize=9)

    ax.axis('off')
    ax.set_xlim([dv_tools.shifted_start(tlim)-0.01*(tlim[1]-tlim[0]),tlim[1]+0.01*(tlim[1]-tlim[0])])
    ax.set_ylim([-0.05,1.05])
    
    if black:
        ax.axvspan(black_co[0], black_co[1], color='gray', zorder=1)

    if grey:
        ax.axvspan(grey_co[0], grey_co[1], color='lightgrey', zorder=1)

    if zoom_area is not None:
        #ax.fill_between(zoom_area, [0,0], [1,1],  color='k', alpha=.2, lw=0)
        #outline is more clear
        if isinstance(zoom_area[0], (list, tuple)):  # multiple regions
            zoom_regions = zoom_area
        else:  # single region
            zoom_regions = [zoom_area]

        for zr in zoom_regions:
            rect = patches.Rectangle((zr[0], 0),    # bottom-left corner (x,y)
                                    zr[1]-zr[0], # width
                                    1,                  # height
                                    linewidth=1.5,
                                    edgecolor='white',#edgecolor='black',
                                    facecolor='none', 
                                    zorder=3)
            ax.add_patch(rect)
    
    return fig, ax


