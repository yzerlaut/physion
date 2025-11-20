# general modules
import pynwb, os, sys, pathlib, itertools, scipy, json
import numpy as np
import matplotlib.pylab as plt

import physion.utils.plot_tools as pt

from physion.utils.files import get_files_with_extension,\
        list_dayfolder, get_TSeries_folders
from physion.assembling.tools import StartTime_to_day_seconds,\
        load_FaceCamera_data
from physion.dataviz.tools import convert_times_to_indices
from physion.utils.binary import BinaryFile
from physion.dataviz.raw import *
from physion.dataviz.imaging import *
from physion.dataviz.camera import *
from physion.dataviz.pupil import *

plt.rcParams['figure.autolayout'] = False

iMap = pt.get_linear_colormap('k','lightgreen')

def layout(show_axes=False):
    """
    default layout for the plot
    """

    AX = {}
    fig = plt.figure(figsize=(8,4))

    AX['axTraces'] = fig.add_axes([0.42, 0.02, 0.54, 0.63])

    AX['axImaging'] = fig.add_axes([0.01, 0.1, 0.3, 0.5])

    AX['axTime'] = fig.add_axes([0.1, 0.01, 0.1, 0.05])

    height0 = 0.68
    AX['axROI1'] = fig.add_axes([0.3, 0.15, 0.1, 0.2])
    AX['axROI2'] = fig.add_axes([0.3, 0.4, 0.1, 0.2])
    height0 += 0.02
    AX['axScreen'] = fig.add_axes([0.01, height0, 0.2, 0.25])
    AX['axRig'] = fig.add_axes([0.25, height0+0.03, 0.17, 0.2])
    AX['axFace'] = fig.add_axes([0.45, height0, 0.2, 0.25])

    AX['axWhisking'] = fig.add_axes([0.67, height0+0.04, 0.15, 0.2])
    AX['cbWhisking'] = fig.add_axes([0.68, height0+0.01, 0.13, 0.02])
    AX['axPupil'] = fig.add_axes([0.84, height0+0.02, 0.15, 0.2])

    AX['cursor'] = AX['axTraces'].plot([0, 1], [0,0], 'k-', lw=3, alpha=.3)[0]
    AX['time'] = AX['axTime'].annotate(' ',
                            (0,0), xycoords='figure fraction', size=8)

    if not show_axes:
        for key in AX:
            if hasattr(AX[key], 'axis'):
                AX[key].axis('off')

    return fig, AX

       
def show_img(img, args,
             key='imaging'):
  
    img -= np.min(img)
    if 'norm_%s'%key in args:
        img /= args['norm_%s'%key]
    if '%s_NL'%key in args:
        img = np.power(img, 1./args['%s_NL'%key])

    if '%s_Lim' % key in args:
        img = img[args['%s_Lim'%key][0]:args['%s_Lim'%key][2],
                  args['%s_Lim'%key][1]:args['%s_Lim'%key][3]]

    return img

def init_imaging(AX, params, data):
    """
    initialize imaging plot based on summary suite2p data
    """

    if 'ophys' in data.nwbfile.processing:

        # full image
        img = getattr(getattr(data.nwbfile.processing['ophys'],
                           'data_interfaces')['Backgrounds_0'],
                           'images')['meanImg'][:]
        params['norm_imaging'] = np.max(img)-np.min(img)

        AX['imgImaging'] = AX['axImaging'].imshow(\
                            show_img(img, params, 'imaging'),
                            vmin=params['imaging_clip'][0]\
                                if 'imaging_clip' in params else 0,
                            vmax=params['imaging_clip'][1]\
                                if 'imaging_clip' in params else 1,
                                    cmap=iMap, origin='lower',
                                        aspect='equal', 
                                            interpolation='none')


        # zoomed ROIs
        for i, roi in enumerate(params['zoomROIs']):
            params['ROI%i_NL'%i] = 1 # NL HERE FOR NOW

            params['ROI%i_extent'%i] = find_roi_extent(data, roi,
                    force_square=True,
                    roi_zoom_factor=params['zoomROIs_factor'][i])

            extent = params['ROI%i_extent'%i]
            img_ROI = img[extent[0]:extent[1],
                                    extent[2]:extent[3]] 
            params['norm_ROI%i'%i] = np.max(img_ROI) 


            AX['imgROI%i' % (i+1)] = \
                    AX['axROI%i' % (i+1)].imshow(
                        show_img(img_ROI, params, 'ROI%i'%i),
                        vmin=params['ROI%i_clip'%(i+1)][0]\
                           if 'ROI%i_clip'%(i+1) in params else 0,
                        vmax=params['ROI%i_clip'%(i+1)][1]\
                           if 'ROI%i_clip'%(i+1) in params else 1,
                        cmap=iMap, aspect='equal', 
                        interpolation='none', origin='lower')
            add_roi_ellipse(data, roi,
                            AX['axROI%i' % (i+1)],
                            size_factor=1.5, roi_lw=1)
    else:
        print("""
            no ophys data in NWB
            """)

def update_imaging(AX, data, params, imagingData, t):

    im_index = dv_tools.convert_time_to_index(t,
                                        data.Fluorescence)
    dS = int(3*params['imaging_temporal_filter'])
    img = imagingData.data[\
        max([im_index-dS,0]):min([im_index+dS+1,imagingData.shape[0]]),
                        :,:].astype(np.uint16).mean(axis=0)
    img = scipy.ndimage.gaussian_filter1d(img,
                            params['imaging_spatial_filter'])
    AX['imgImaging'].set_array(\
            show_img(img, params, 'imaging'))

    for n, roi in enumerate(params['zoomROIs']):
        extent = params['ROI%i_extent'%n]
        img_ROI = img[extent[0]:extent[1], extent[2]:extent[3]] 
        AX['imgROI%i' % (n+1)].set_array(\
                    show_img(img_ROI, params,
                              'ROI%i'%n))


def init_screen(AX, data):

    # screen inset
    AX['imgScreen'] = AX['axScreen'].imshow(\
                            0*data.visual_stim.x+0.5,
                        extent=(0, data.visual_stim.x.shape[0],
                                0, data.visual_stim.x.shape[1]),
                        cmap='gray',
                        vmin=0, vmax=1,
                        origin='lower',
                        aspect='equal')


def update_screen(AX, data, t):

    # visual stim
    iEp = data.find_episode_from_time(t)
    if iEp==-1:
        AX['imgScreen'].set_array(data.visual_stim.x*0+0.5)
    else:
        tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
        AX['imgScreen'].set_array(data.visual_stim.get_image(iEp,
                                        time_from_episode_start=t-tEp))
        # data.visual_stim.update_frame(iEp, AX['imgScreen'],
        #                                 time_from_episode_start=t-tEp)

def get_camera_img(camera, t=0):

    index = np.argmin((camera.times-t)**2)

    return np.array(camera.get(index).T, dtype=float)/255.

def init_camera(AX, params, camera, name='Face'):

    if camera is not None:
        
        img= get_camera_img(camera)

        params['norm_%s' % name] = np.max(img)
        AX['img%s' % name] = AX['ax%s' % name].imshow(\
                        show_img(img, params, name),
            vmin=params['%s_clip' % name][0] if '%s_clip'%name in params else 0,
            vmax=params['%s_clip' % name][1] if '%s_clip'%name in params else 1,
                        cmap='gray')

    else:

        AX['img%s' % name] = AX['ax%s' % name].imshow(np.zeros(2,2))

def update_camera(AX, params, camera, t=0, name='Face'):

    img= get_camera_img(camera, t)

    AX['imgRig'].set_array(show_img(img, params, 'Rig'))

def get_pupil_center(index, data):
    coords = []
    for key in ['cx', 'cy']:
        coords.append(\
            data.nwbfile.processing['Pupil'].data_interfaces[key].data[index]*data.FaceCamera_mm_to_pix)
    return coords

def get_pupil_fit(index, data):

    coords = []

    for key in ['cx', 'cy', 'sx', 'sy']:
        coords.append(data.FaceCamera_mm_to_pix*\
            data.nwbfile.processing['Pupil'].data_interfaces[key].data[index])

    if 'angle' in data.nwbfile.processing['Pupil'].data_interfaces:
        coords.append(\
            data.nwbfile.processing['Pupil'].data_interfaces['angle'].data[index])
    else:
        coords.append(0)

    return process.ellipse_coords(*coords, transpose=False)
    

def init_pupil(AX, data, params, faceCamera):

    imgFace = faceCamera.get(0).T

    x, y = np.meshgrid(np.arange(0,imgFace.shape[0]),
                    np.arange(0,imgFace.shape[1]), 
                    indexing='ij')

    pupil_cond = (y>=data.pupil_ROI['xmin']) &\
                (y<=data.pupil_ROI['xmax']) &\
                (x>=data.pupil_ROI['ymin']) &\
                (x<=data.pupil_ROI['ymax'])

    pupil_shape = len(np.unique(x[pupil_cond])),\
                            len(np.unique(y[pupil_cond]))
    AX['imgPupil'] = AX['axPupil'].imshow(\
            imgFace[pupil_cond].reshape(*pupil_shape), 
            cmap='gray')

    params['pupil_cond'] = pupil_cond
    params['pupil_shape'] = pupil_shape
    pupil_fit = get_pupil_fit(0, data)
    AX['pupil_fit'], = AX['axPupil'].plot(\
                            pupil_fit[0], pupil_fit[1],
                            '.', markersize=1, color='red')

    pupil_center = get_pupil_center(0, data)
    AX['pupil_center'], = AX['axPupil'].plot(\
                    [pupil_center[0]], [pupil_center[1]], '.',
                    markersize=5, color='orange')

def update_pupil(AX, data, params, faceCamera, t):

    index = np.argmin((faceCamera.times-t)**2)

    AX['imgPupil'].set_array(
            faceCamera.get(index).T[params['pupil_cond']].reshape(*params['pupil_shape']))

    pupil_fit = get_pupil_fit(index, data)
    AX['pupil_fit'].set_data(pupil_fit[0], pupil_fit[1])

    pupil_center = get_pupil_center(index, data)
    AX['pupil_center'].set_data(\
                    [pupil_center[0]], [pupil_center[1]])

def init_whisking(AX, data, params, faceCamera):

    imgFace = faceCamera.get(0).T

    x, y = np.meshgrid(np.arange(0,imgFace.shape[0]),
                    np.arange(0,imgFace.shape[1]), 
                    indexing='ij')

    whisking_cond = (x>=data.FaceMotion_ROI[0]) &\
                (x<=(data.FaceMotion_ROI[0]+data.FaceMotion_ROI[2])) &\
                (y>=data.FaceMotion_ROI[1]) &\
                (y<=(data.FaceMotion_ROI[1]+data.FaceMotion_ROI[3]))

    whisking_shape = len(np.unique(x[whisking_cond])),\
                            len(np.unique(y[whisking_cond]))
    params['whisking_cond'] = whisking_cond
    params['whisking_shape'] = whisking_shape

    img1 = faceCamera.get(1).astype(float).T
    img = faceCamera.get(0).astype(float).T

    new_img = (img1-img)[whisking_cond].reshape(\
                                        *whisking_shape)
    AX['imgWhisking'] = AX['axWhisking'].imshow(\
                        new_img,
                        vmin=-255/5., vmax=255/5.,
                        cmap=plt.cm.BrBG)

    plt.colorbar(AX['imgWhisking'], 
                orientation='horizontal',
                cax=AX['cbWhisking'])

def update_whisking(AX, data, params, faceCamera, t):

    index = np.argmin((faceCamera.times-t)**2)

    img1 = faceCamera.get(index+1).astype(float).T
    img = faceCamera.get(index).astype(float).T

    AX['imgWhisking'].set_array(
            (img1-img)[params['whisking_cond']].reshape(*params['whisking_shape']))

def plot_traces(AX, params, data):

    #   ----  filling time plot

    # visual stim
    if 'visual_stim' in params['fractions']:
        add_VisualStim(data, params['tlim'], AX['axTraces'], 
                       fig_fraction=params['fractions']['visual_stim_start'], 
                       with_screen_inset=bool(params['with_screen_inset']),
                       name='')

    # photodiode
    if 'photodiode' in params['fractions']:
        add_Photodiode(data, params['tlim'], AX['axTraces'], 
            fig_fraction_start=params['fractions']['photodiode_start'], 
            fig_fraction=params['fractions']['photodiode'], name='')

    # locomotion
    if 'running' in params['fractions']:
        add_Locomotion(data, params['tlim'], AX['axTraces'], 
                    fig_fraction_start=params['fractions']['running_start'], 
                    fig_fraction=params['fractions']['running'], 
                    scale_side='right', subsampling=1,
                    name='')

    # whisking 
    if 'whisking' in params['fractions']:
        add_FaceMotion(data, params['tlim'], AX['axTraces'], 
                fig_fraction_start=params['fractions']['whisking_start'], 
                fig_fraction=params['fractions']['whisking'], 
                scale_side='right', subsampling=1, name='')

    # gaze 
    if 'gaze' in params['fractions']:
        add_GazeMovement(data, params['tlim'], AX['axTraces'], 
                fig_fraction_start=params['fractions']['gaze_start'], 
                fig_fraction=params['fractions']['gaze'], 
                scale_side='right', name='')

    # pupil 
    if 'pupil' in params['fractions']:
        add_Pupil(data, params['tlim'], AX['axTraces'], 
                    fig_fraction_start=params['fractions']['pupil_start'], 
                    fig_fraction=params['fractions']['pupil'], 
                    scale_side='right', subsampling=1, name='')

    # rois 
    if 'ophys' in data.nwbfile.processing:
        data.build_rawFluo()
        if 'dFoF' in params['trace_quantity']:
            data.build_dFoF(smoothing=params['dFoF_smoothing'])
        add_CaImaging(data, params['tlim'], AX['axTraces'], 
                      subsampling=1,
                      subquantity=params['trace_quantity'],
                      roiIndices=params['ROIs'], 
                      fig_fraction_start=params['fractions']['rois_start'], 
                      fig_fraction=params['fractions']['rois'], 
                      scale_side='right',
                      name='', annotation_side='')

        # raster 
        if 'raster' in params['fractions']:
            data.build_dFoF(smoothing=2)
            add_CaImagingRaster(data,params['tlim'],AX['axTraces'], 
                        # subquantity=params['trace_quantity'],
                        subsampling=1,
                        subquantity='dFoF',
                        normalization='per-line',
                        bar_inset_start=1.02,
                        fig_fraction_start=params['fractions']['raster_start'], 
                        fig_fraction=params['fractions']['raster'], 
                        name='')

    if params['Tbar']>0:
        AX['axTraces'].plot(params['Tbar']*np.arange(2)+params['tlim'][0],
                            params['Tbar_loc']*np.ones(2), 'k-', lw=1)
        AX['axTraces'].annotate('%is' % params['Tbar'],
                                (params['tlim'][0], 1.005*params['Tbar_loc']),
                                 ha='left', fontsize=8,)
    # AX['axTraces'].set_xlim(params['tlim'])
    AX['axTraces'].set_xlim([params['tlim'][0], AX['axTraces'].get_xlim()[1]])
    AX['axTraces'].set_ylim([-0.01, 1.01])

def update_timer(AX, time):
    AX['cursor'].set_data(np.ones(2)*time, np.arange(2))
    AX['time'].set_text('     t=%.1fs\n' % time)

def load_Imaging(metadata):
    metadata['raw_Imaging_folder'] = params['raw_Imaging_folder']

def fill_sheet_with_datafiles(nwbfile):
    """
    """

if __name__=='__main__':

    import physion, tempfile

    if sys.argv[-1]=='show-layout':
        # just showing the current figure layout
        fig, AX = layout(show_axes=True)
        plt.show()

    elif sys.argv[-1]=='generate-template':
        # we write a default params file
        with open('visualization_params.json', 'w') as f:
            f.write(default_params)

        print("""
            wrote a default parameter file as:
                    ./visualization_params.json
              modify this file to specify the visualization properties
                and run:
                    python -m physion.dataviz.snapshot ./visualization_params.json
              """)
    else:
        fn = tempfile.mktemp(suffix='.json')
        with open(fn, 'w') as f:
            f.write(default_params)
        with open(fn, 'r') as f:
            params = json.load(f)

        data = physion.analysis.read_NWB.Data(os.path.expanduser(params['nwbfile']),
                                              with_visual_stim=True)
        
        # print('tlim: %s' % data.tlim)

        fig, AX, metadata = draw_figure(params, data)    

        plt.show()
        # root_path = os.path.dirname(args.datafile)
        # subfolder = os.path.basename(\
        #         args.datafile).replace('.nwb','')[-8:]



    """
    if ('.nwb' in args.datafile) and os.path.isfile(args.datafile):


        data = physion.analysis.read_NWB.Data(args.datafile,
                                              with_visual_stim=True)
        print('tlim: %s' % data.tlim)
        root_path = os.path.dirname(args.datafile)
        subfolder = os.path.basename(\
                args.datafile).replace('.nwb','')[-8:]

         # "raw_Behavior_folder"
        if os.path.isdir(os.path.join(root_path, subfolder)):
            args.raw_Behavior_folder = os.path.join(root_path,
                                                    subfolder)
        else:
            print(os.path.join(root_path, subfolder), 'not found')

         # "raw_Imaging_folder"
        if os.path.isdir(os.path.join(root_path,data.TSeries_folder)):
            args.raw_Imaging_folder = os.path.join(root_path,
                                                data.TSeries_folder)
        else:
            print(os.path.join(root_path, data.TSeries_folder),
                  'not found')

        for key in params:
            if not hasattr(args, key):
                setattr(args, key, params[key])

        fig, AX, metadata = draw_figure(vars(args), data)    

        plt.show()

    else:
        print('')
        print(' provide either a nwbfile or a snapshot.py file as argument')
        print('')


    """
