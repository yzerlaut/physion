# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from PIL import Image

import physion.utils.plot_tools as pt

from physion.pupil import roi, process
from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders
from physion.assembling.tools import StartTime_to_day_seconds, load_FaceCamera_data
from physion.dataviz.tools import convert_times_to_indices
from physion.assembling.IO.binary import BinaryFile
from physion.dataviz.raw import *
from physion.dataviz.imaging import *

plt.rcParams['figure.autolayout'] = False

from physion.dataviz.snapshot import *


def draw_movie(args, data,
               Ndiscret=100):

    fig, AX = draw_figure(args, data)


    times = np.linspace(args['tlim'][0], args['tlim'][1], 
                        args['Ndiscret'])


    """
    if 'ophys' in data.nwbfile.processing:

        # full image
        max_proj = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'][:]
        max_proj_scaled = np.power(max_proj/max_proj.max(),
                                   1/args.imaging_NL)

        AX['imgImaging'] = AX['axImaging'].imshow(max_proj_scaled, 
                    vmin=0, vmax=1, cmap=iMap, origin='lower',
                    aspect='equal', interpolation='none')

        AX['axImaging'].annotate(' n=%i rois' % data.iscell.sum(),
                                  (0,0), color='w', fontsize=8,
                                  xycoords='axes fraction')

        # ROIs
        extents, max_projs = [], []
        for i, roi in enumerate(args.ROIs):
            extents.append(find_roi_extent(data, args.ROIs[i],
                                           roi_zoom_factor=5))
            max_projs.append(\
                  data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'][:][extents[i][0]:extents[i][1], extents[i][2]:extents[i][3]])
            # max_proj_scaled1 = (max_proj-max_proj.min())/(max_proj.max()-max_proj.min())
            # max_proj_scaled1 = np.power(max_proj_scaled1, 1/args.imaging_NL)

            AX['imgROI%i' % (i+1)] = \
                    AX['axROI%i' % (i+1)].imshow(max_projs[i],
                            vmin=0, vmax=max_projs[i].max(), 
                            cmap=iMap, extent=extents[i],
                            aspect='equal', interpolation='none', 
                            origin='lower')
            add_roi_ellipse(data, args.ROIs[i], 
                            AX['axROI%i' % (i+1)],
                            size_factor=1.5, roi_lw=1)

    AX['axWhisking'].annotate('whisker-pad\nmotion frames',
                              (0.,0.5), va='center', ha='right', rotation=90,
                              fontsize=7, xycoords='axes fraction')

    AX['axPupil'].annotate('ellipse\nfitting',
                           (0.,0.5), va='center', ha='right', rotation=90,
                           fontsize=7, xycoords='axes fraction')

    t0 = times[0]

    # setup drawing
    # time = AX['axTime'].annotate(' ', (0,0), xycoords='figure fraction', size=9)
    time = AX['axTime'].annotate('     t=%.1fs\n' % times[0], (0,0), xycoords='figure fraction', size=9)

    # screen inset
    AX['imgScreen'] = data.visual_stim.show_frame(0,
                                                  ax=AX['axScreen'],
                                                  return_img=True,
                                                  label=None)

    # Calcium Imaging
    if metadata['raw_imaging_folder']!='':
        
        Ly, Lx = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'].shape
        Ca_data = BinaryFile(Ly=Ly, Lx=Lx,
                             read_filename=os.path.join(metadata['raw_imaging_folder'],
                                                       'suite2p', 'plane0','data.bin'))
        i1, i2 = convert_times_to_indices(times[0], times[1],
                                          data.Fluorescence)

        imaging_scale = Ca_data.data[i1:i2,:,:].min(), Ca_data.data[i1:i2,:,:].max()

        imaging_scales = []

        for n in range(len(args.ROIs)):
            imaging_scales.append(\
                    (Ca_data.data[i1:i2,
                                 extents[n][0]:extents[n][1],
                                 extents[n][2]:extents[n][3]].min(),\
                    (Ca_data.data[i1:i2,
                                 extents[n][0]:extents[n][1],
                                 extents[n][2]:extents[n][3]].max())))

    else:

        Ca_data = None


    # Face Camera
    if metadata['raw_vis_folder']!='':

        load_NIdaq(metadata)

        loadCameraData(metadata)

        # Rig Image
        img = np.load(metadata['raw_Rig_FILES'][0])
        AX['imgRig'] = AX['axRig'].imshow(imgRig_process(img,args),
                                    vmin=0, vmax=1, cmap='gray')

        # Pupil Image
        img = np.load(metadata['raw_Face_FILES'][0])
        AX['imgFace'] = AX['axFace'].imshow(imgFace_process(img,args),
                                            vmin=0, vmax=1, cmap='gray')

        # pupil
        x, y = np.meshgrid(np.arange(0,img.shape[0]), np.arange(0,img.shape[1]), indexing='ij')
        pupil_cond = (x>=metadata['pupil_xmin']) & (x<=metadata['pupil_xmax']) & (y>=metadata['pupil_ymin']) & (y<=metadata['pupil_ymax'])
        pupil_shape = len(np.unique(x[pupil_cond])), len(np.unique(y[pupil_cond]))
        uh23pupil_shape = metadata['pupil_xmax']-metadata['pupil_xmin']+1, metadata['pupil_ymax']-metadata['pupil_ymin']+1
        AX['imgPupil'] = AX['axPupil'].imshow(img[pupil_cond].reshape(*pupil_shape), cmap='gray')
        pupil_fit = get_pupil_fit(0, data, metadata)
        AX['pupil_fit'], = AX['axPupil'].plot(pupil_fit[0], pupil_fit[1], '.', markersize=2, color='red')
        pupil_center = get_pupil_center(0, data, metadata)
        AX['pupil_center'], = AX['axPupil'].plot([pupil_center[1]], [pupil_center[0]], '.', markersize=8, color='orange')

        # whisking
        whisking_cond = (x>=metadata['whisking_ROI'][0]) & (x<=(metadata['whisking_ROI'][0]+metadata['whisking_ROI'][2])) &\
                (y>=metadata['whisking_ROI'][1]) & (y<=(metadata['whisking_ROI'][1]+metadata['whisking_ROI'][3]))
        whisking_shape = len(np.unique(x[whisking_cond])), len(np.unique(y[whisking_cond]))
        img1 = np.load(metadata['raw_Face_FILES'][1])

        new_img = (img1-img)[whisking_cond].reshape(*whisking_shape)
        AX['imgWhisking'] = AX['axWhisking'].imshow(new_img,
                                                      vmin=-255/4, vmax=255+255/4,
                                                      cmap=plt.cm.BrBG)


    # time cursor
    cursor, = AX['axTraces'].plot(\
         np.ones(2)*times[0], np.arange(2),
         'k-')#color='grey', lw=3, alpha=.3) 

    #   ----  filling time plot

    # photodiode and visual stim
    if not args.no_visual:
        add_VisualStim(data, args.tlim, AX['axTraces'], 
                       fig_fraction=2., with_screen_inset=True,
                       name='')
        # add_Photodiode(data, args.tlim, AX['axTraces'], 
                       # fig_fraction_start=fractions['photodiode_start'], 
                       # fig_fraction=fractions['photodiode'], name='')
        # AX['axTraces'].annotate('photodiode', (-0.01, fractions['photodiode_start']),
                # ha='right', va='bottom', color='grey', fontsize=8, xycoords='axes fraction')


    # locomotion
    add_Locomotion(data, args.tlim, AX['axTraces'], 
                        fig_fraction_start=fractions['running_start'], 
                        fig_fraction=fractions['running'], 
                        scale_side='right', subsampling=1,
                        name='')
    AX['axTraces'].annotate('running \nspeed \n ', (-0.01, fractions['running_start']),
            ha='right', va='bottom', color='#1f77b4', fontsize=8, xycoords='axes fraction')

    # whisking 
    add_FaceMotion(data, args.tlim, AX['axTraces'], 
                   fig_fraction_start=fractions['whisking_start'], 
                   fig_fraction=fractions['whisking'], 
                   scale_side='right', subsampling=1,
                   name='')
    AX['axTraces'].annotate('whisking \n', (-0.01, fractions['whisking_start']),
            ha='right', va='bottom', color='purple', fontsize=8, xycoords='axes fraction')

    # gaze 
    # add_GazeMovement(data, args.tlim, AX['axTraces'], 
                        # fig_fraction_start=fractions['gaze_start'], 
                        # fig_fraction=fractions['gaze'], 
                  # scale_side='right',
                        # name='')
    # AX['axTraces'].annotate('gaze \nmov. ', (-0.01, fractions['gaze_start']),
            # ha='right', va='bottom', color='orange', fontsize=8, xycoords='axes fraction')

    # pupil 
    add_Pupil(data, args.tlim, AX['axTraces'], 
                        fig_fraction_start=fractions['pupil_start'], 
                        fig_fraction=fractions['pupil'], 
                        scale_side='right', subsampling=1,
                        name='')
    AX['axTraces'].annotate('pupil \ndiam. ', (-0.01, fractions['pupil_start']),
            ha='right', va='bottom', color='red', fontsize=8, xycoords='axes fraction')

    # rois 
    if 'ophys' in data.nwbfile.processing:
        add_CaImaging(data, args.tlim, AX['axTraces'], 
                      subquantity='dFoF',
                      roiIndices=args.ROIs, 
                      fig_fraction_start=fractions['rois_start'], 
                      fig_fraction=fractions['rois'], 
                      scale_side='right',
                      name='', annotation_side='left')
        AX['axTraces'].annotate('fluorescence', (-0.1,
                    fractions['rois_start']+fractions['rois']/2.),
                ha='right', va='center', color='green', rotation=90, xycoords='axes fraction')

        # raster 
        add_CaImagingRaster(data, args.tlim, AX['axTraces'], 
                    subquantity='dFoF', 
                    normalization='per-line',
                    fig_fraction_start=fractions['raster_start'], 
                    fig_fraction=fractions['raster'], 
                    name='')
    else:
        AX['dFoFscale_ax'], AX['dFoFscale_cb'] = None, None

    if args.Tbar>0:
        AX['axTraces'].plot(args.Tbar*np.arange(2)+times[0],
                                args.Tbar_loc*np.ones(2), 'k-', lw=1)
        AX['axTraces'].annotate('%is' % args.Tbar,
                                    (times[0],args.Tbar_loc), ha='left')

    AX['axTraces'].axis('off')
    AX['axTraces'].set_xlim([times[0], dv_tools.shifted_stop(args.tlim)])
    AX['axTraces'].set_ylim([-0.01, 1.01])

    def update(i=0):

        if 'raw_Rig_times' in metadata:
            # Rig camera
            camera_index = np.argmin((metadata['raw_Rig_times']-times[i])**2)
            img = np.load(metadata['raw_Rig_FILES'][camera_index])
            AX['imgRig'].set_array(imgRig_process(img, args))

        if 'raw_Face_times' in metadata:
            # Face camera
            camera_index = np.argmin((metadata['raw_Face_times']-times[i])**2)
            img = np.load(metadata['raw_Face_FILES'][camera_index])
            AX['imgFace'].set_array(imgFace_process(img, args))
            # pupil
            AX['imgPupil'].set_array(img[pupil_cond].reshape(*pupil_shape))
            pupil_fit = get_pupil_fit(camera_index, data, metadata)
            AX['pupil_fit'].set_data(pupil_fit[1], pupil_fit[0])
            pupil_center = get_pupil_center(camera_index, data, metadata)
            AX['pupil_center'].set_data([pupil_center[1]], [pupil_center[0]])
            # whisking
            img1 = np.load(metadata['raw_Face_FILES'][camera_index+1])
            AX['imgWhisking'].set_array((img1-img)[whisking_cond].reshape(*whisking_shape))

        # imaging
        if (i in [0,len(times)-1]) or (Ca_data is None):
            AX['imgImaging'].set_array(max_proj_scaled)
            for n in range(len(args.ROIs)):
                AX['imgROI%i' % (n+1)].set_array(max_projs[n])
        else:
            im_index = dv_tools.convert_time_to_index(times[i], data.Fluorescence)
            img = Ca_data.data[im_index,:,:].astype(np.uint16)
            img = np.power(img/np.max(max_proj), 1/args.imaging_NL)
            AX['imgImaging'].set_array(img)

            for n in range(len(args.ROIs)):
                imgN = Ca_data.data[im_index,
                                    extents[n][0]:extents[n][1],
                                    extents[n][2]:extents[n][3]]
                imgN = (imgN-imaging_scales[n][0])/(imaging_scales[n][1]-imaging_scales[n][0])
                imgN = np.power(imgN, 1/args.imaging_NL)
                AX['imgROI%i' % (n+1)].set_array(imgN)

        # visual stim
        if not args.no_visual:
            iEp = data.find_episode_from_time(times[i])
            if iEp==-1:
                AX['imgScreen'].set_array(data.visual_stim.x*0+0.5)
            else:
                tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
                data.visual_stim.update_frame(iEp, AX['imgScreen'],
                                              time_from_episode_start=times[i]-tEp)
        cursor.set_data(np.ones(2)*times[i], np.arange(2))
        # time
        time.set_text('     t=%.1fs\n' % times[i])
        
        return [cursor, time, AX['imgScreen'], 
                AX['imgRig'], AX['imgFace'],
                AX['imgPupil'], AX['imgWhisking'], 
                AX['pupil_fit'], AX['pupil_center'],
                AX['imgImaging'], 
                AX['imgROI1'], AX['imgROI2'], AX['imgROI3']]
       

    if args.export or not args.snapshot:
        ani = animation.FuncAnimation(fig, 
                                      update,
                                      np.arange(len(times)),
                                      init_func=update,
                                      interval=100,
                                      blit=True)

    else:
        return fig, AX, None 
    """
    ani=None
    return fig, AX, ani

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)

    parser.add_argument("--fps", 
                        type=int, default=20)
    parser.add_argument("--duration", 
                        type=float, default=0, help='video duration')
    parser.add_argument("--dpi", 
                        type=int, default=100, help='video duration')

    parser.add_argument("--export", action="store_true")

    args = parser.parse_args()

    if args.duration>0:
        args.Ndiscret = int(args.duration*args.fps)
    else:
        args.Ndiscret = 10


    args = vars(args)

    if os.path.isfile(args['datafile']):

        with open(args['datafile']) as f:
            string_params = f.read()
            exec(string_params)

        params['Ndiscret'] = 10
        params['datafile'] = params['nwbfile']
        data = physion.analysis.read_NWB.Data(params['datafile'],
                                              with_visual_stim=True)

        fig, AX, ani = draw_movie(params, data)

        if args['export']:
            print('writing video [...]')
            writer = animation.writers['ffmpeg'](fps=args.fps)
            ani.save(args.output, writer=writer, dpi=args.dpi)

        else:
            plt.show()

        plt.show()

    else:
        print('')
        print(' provide either a movie.py file as argument')
        print('')




