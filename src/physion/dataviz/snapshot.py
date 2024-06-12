# general modules
import pynwb, os, sys, pathlib, itertools, scipy
import numpy as np
import matplotlib.pylab as plt

import physion.utils.plot_tools as pt

from physion.utils.files import get_files_with_extension,\
        list_dayfolder, get_TSeries_folders
from physion.assembling.tools import StartTime_to_day_seconds,\
        load_FaceCamera_data
from physion.dataviz.tools import convert_times_to_indices
from physion.assembling.IO.binary import BinaryFile
from physion.dataviz.raw import *
from physion.dataviz.imaging import *
from physion.dataviz.camera import *
from physion.dataviz.pupil import *

plt.rcParams['figure.autolayout'] = False

iMap = pt.get_linear_colormap('k','lightgreen')

string_params = """
params = {

    ############################################
    ###         DATAFILE         ###############
    ############################################
    'nwbfile':'-',
    'raw_Behavior_folder':'',
    'raw_Imaging_folder':'',

    ############################################
    ###         VIEW OPTIONS     ###############
    ############################################
    'tlim':[20,80],
    'Ndiscret':100, # for movie only

    # imaging
    'imaging_temporal_filter':3.0,
    'imaging_spatial_filter':0.8,
    'imaging_NL':3,
    'imaging_clip':[0.3, 0.9],
    'trace_quantity':'rawFluo',
    'dFoF_smoothing':0.1,
    # ROIs zoom
    'zoomROIs_factor':[3.0,2.5],

    # FaceCamera
    'Face_Lim':[0, 0, 10000, 10000],
    'Face_clip':[0.3,1.],
    'Face_NL':5,
    # RigCamera
    'Rig_Lim':[100, 100, 470, 750],
    'Rig_NL':2,

    ############################################
    ###      ANNOTATIONS         ###############
    ############################################
    'Tbar':2, 'Tbar_loc':1.0,
    'with_screen_inset':False,

    ############################################
    ###       LAYOUT OPTIONS     ###############
    ############################################
    'ROIs':range(5),
    'fractions': {'running':0.1, 'running_start':0.89,
                  'whisking':0.1, 'whisking_start':0.78,
                  'gaze':0.08, 'gaze_start':0.7,
                  'pupil':0.15, 'pupil_start':0.55,
                  'rois':0.29, 'rois_start':0.29,
                  'visual_stim':2, 'visual_stim_start':2.,
                  'raster':0.28, 'raster_start':0.},
}
"""


def layout(args):

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

    if (not 'layout' in args) or (not args['layout']):
        for key in AX:
            AX[key].axis('off')

    AX['cursor'] = AX['axTraces'].plot(args['tlim'][0]*np.ones(2), 
                                       [0,0], 'k-', lw=3, alpha=.3)[0]
    AX['time'] = AX['axTime'].annotate(' ',
                            (0,0), xycoords='figure fraction', size=8)

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

def draw_figure(args, data):

    fig, AX = layout(args)

    metadata = dict(data.metadata)
    metadata['raw_Behavior_folder'] = args['raw_Behavior_folder']
    metadata['raw_Imaging_folder'] = args['raw_Imaging_folder']


    if 'ophys' in data.nwbfile.processing:

        # full image
        img = getattr(getattr(data.nwbfile.processing['ophys'],
                           'data_interfaces')['Backgrounds_0'],
                           'images')['meanImg'][:]
        args['norm_imaging'] = np.max(img)-np.min(img)

        AX['imgImaging'] = AX['axImaging'].imshow(\
                            show_img(img, args, 'imaging'),
                            vmin=args['imaging_clip'][0]\
                                if 'imaging_clip' in args else 0,
                            vmax=args['imaging_clip'][1]\
                                if 'imaging_clip' in args else 1,
                                    cmap=iMap, origin='lower',
                                        aspect='equal', 
                                            interpolation='none')


        # zoomed ROIs
        for i, roi in enumerate(args['zoomROIs']):
            args['ROI%i_NL'%i] = 1 # NL HERE FOR NOW

            args['ROI%i_extent'%i] = find_roi_extent(data, roi,
                    force_square=True,
                    roi_zoom_factor=args['zoomROIs_factor'][i])

            extent = args['ROI%i_extent'%i]
            img_ROI = img[extent[0]:extent[1],
                                    extent[2]:extent[3]] 
            args['norm_ROI%i'%i] = np.max(img_ROI) 


            AX['imgROI%i' % (i+1)] = \
                    AX['axROI%i' % (i+1)].imshow(
                        show_img(img_ROI, args, 'ROI%i'%i),
                        vmin=args['ROI%i_clip'%(i+1)][0]\
                           if 'ROI%i_clip'%(i+1) in args else 0,
                        vmax=args['ROI%i_clip'%(i+1)][1]\
                           if 'ROI%i_clip'%(i+1) in args else 1,
                        cmap=iMap, aspect='equal', 
                        interpolation='none', origin='lower')
            add_roi_ellipse(data, roi,
                            AX['axROI%i' % (i+1)],
                            size_factor=1.5, roi_lw=1)


    # screen inset
    AX['imgScreen'] = data.visual_stim.show_frame(0,
                                                  ax=AX['axScreen'],
                                                  return_img=True,
                                                  label=None)

    # Face Camera
    if metadata['raw_Behavior_folder']!='':

        loadCameraData(metadata, metadata['raw_Behavior_folder'])

        # Rig Image
        imgRig = np.load(\
                metadata['raw_Rig_FILES'][0]).astype(float)
        args['norm_Rig'] = np.max(imgRig)
        AX['imgRig'] = AX['axRig'].imshow(\
                        show_img(imgRig, args, 'Rig'),
            vmin=args['Rig_clip'][0] if 'Rig_clip' in args else 0,
            vmax=args['Rig_clip'][1] if 'Rig_clip' in args else 1,
                        cmap='gray')

        # Face Image
        imgFace = np.load(\
                metadata['raw_Face_FILES'][0]).astype(float)
        args['norm_Face'] = np.max(imgFace)
        AX['imgFace'] = AX['axFace'].imshow(\
                        show_img(imgFace, args, 'Face'),
         vmin=args['Face_clip'][0] if 'Face_clip' in args else 0,
         vmax=args['Face_clip'][1] if 'Face_clip' in args else 1,
                        cmap='gray')

        # pupil
        if 'pupil' in args['fractions']:
            x, y = np.meshgrid(np.arange(0,imgFace.shape[0]),
                               np.arange(0,imgFace.shape[1]), 
                               indexing='ij')
            pupil_cond = (y>=metadata['pupil_xmin']) &\
                         (y<=metadata['pupil_xmax']) &\
                         (x>=metadata['pupil_ymin']) &\
                         (x<=metadata['pupil_ymax'])
            pupil_shape = len(np.unique(x[pupil_cond])),\
                                    len(np.unique(y[pupil_cond]))
            AX['imgPupil'] = AX['axPupil'].imshow(\
                    imgFace[pupil_cond].reshape(*pupil_shape), cmap='gray')
            metadata['pupil_cond'] = pupil_cond
            metadata['pupil_shape'] = pupil_shape
            pupil_fit = get_pupil_fit(0, data, metadata)
            AX['pupil_fit'], = AX['axPupil'].plot(pupil_fit[0],
                                                  pupil_fit[1],
                                '.', markersize=1, color='red')

        AX['pupil_center'] = None
        if 'gaze' in args['fractions']:
            pupil_center = get_pupil_center(0, data, metadata)
            AX['pupil_center'], = AX['axPupil'].plot(\
                            [pupil_center[0]], [pupil_center[1]], '.',
                            markersize=5, color='orange')

        # whisking
        if 'whisking' in args['fractions']:
            whisking_cond = (x>=metadata['whisking_ROI'][0]) &\
              (x<=(metadata['whisking_ROI'][0]+metadata['whisking_ROI'][2])) &\
              (y>=metadata['whisking_ROI'][1]) &\
              (y<=(metadata['whisking_ROI'][1]+metadata['whisking_ROI'][3]))
            whisking_shape = len(np.unique(x[whisking_cond])),\
                                    len(np.unique(y[whisking_cond]))
            metadata['whisking_cond'] = whisking_cond
            metadata['whisking_shape'] = whisking_shape

            img1 = np.load(\
                    metadata['raw_Face_FILES'][1]).astype(float)
            img = np.load(\
                    metadata['raw_Face_FILES'][0]).astype(float)
            new_img = (img1-img)[whisking_cond].reshape(\
                                                *whisking_shape)
            AX['imgWhisking'] = AX['axWhisking'].imshow(\
                                new_img,
                                vmin=-255/5., vmax=255/5.,
                                cmap=plt.cm.BrBG)
            plt.colorbar(AX['imgWhisking'], 
                         orientation='horizontal',
                         cax=AX['cbWhisking'])


    #   ----  filling time plot

    # visual stim
    if 'visual_stim' in args['fractions']:
        add_VisualStim(data, args['tlim'], AX['axTraces'], 
                       fig_fraction=args['fractions']['visual_stim_start'], 
                       with_screen_inset=bool(args['with_screen_inset']),
                       name='')

    # photodiode
    if 'photodiode' in args['fractions']:
        add_Photodiode(data, args['tlim'], AX['axTraces'], 
            fig_fraction_start=args['fractions']['photodiode_start'], 
            fig_fraction=args['fractions']['photodiode'], name='')

    # locomotion
    if 'running' in args['fractions']:
        add_Locomotion(data, args['tlim'], AX['axTraces'], 
                    fig_fraction_start=args['fractions']['running_start'], 
                    fig_fraction=args['fractions']['running'], 
                    scale_side='right', subsampling=1,
                    name='')

    # whisking 
    if 'whisking' in args['fractions']:
        add_FaceMotion(data, args['tlim'], AX['axTraces'], 
                fig_fraction_start=args['fractions']['whisking_start'], 
                fig_fraction=args['fractions']['whisking'], 
                scale_side='right', subsampling=1, name='')

    # gaze 
    if 'gaze' in args['fractions']:
        add_GazeMovement(data, args['tlim'], AX['axTraces'], 
                fig_fraction_start=args['fractions']['gaze_start'], 
                fig_fraction=args['fractions']['gaze'], 
                scale_side='right', name='')

    # pupil 
    if 'pupil' in args['fractions']:
        add_Pupil(data, args['tlim'], AX['axTraces'], 
                    fig_fraction_start=args['fractions']['pupil_start'], 
                    fig_fraction=args['fractions']['pupil'], 
                    scale_side='right', subsampling=1, name='')

    # rois 
    if 'ophys' in data.nwbfile.processing:
        data.build_rawFluo()
        if 'dFoF' in args['trace_quantity']:
            data.build_dFoF(smoothing=args['dFoF_smoothing'])
        add_CaImaging(data, args['tlim'], AX['axTraces'], 
                      subsampling=1,
                      subquantity=args['trace_quantity'],
                      roiIndices=args['ROIs'], 
                      fig_fraction_start=args['fractions']['rois_start'], 
                      fig_fraction=args['fractions']['rois'], 
                      scale_side='right',
                      name='', annotation_side='')

        # raster 
        if 'raster' in args['fractions']:
            data.build_dFoF(smoothing=2)
            add_CaImagingRaster(data,args['tlim'],AX['axTraces'], 
                        # subquantity=args['trace_quantity'],
                        subsampling=1,
                        subquantity='dFoF',
                        normalization='per-line',
                        bar_inset_start=1.02,
                        fig_fraction_start=args['fractions']['raster_start'], 
                        fig_fraction=args['fractions']['raster'], 
                        name='')

    if args['Tbar']>0:
        AX['axTraces'].plot(args['Tbar']*np.arange(2)+args['tlim'][0],
                            args['Tbar_loc']*np.ones(2), 'k-', lw=1)
        AX['axTraces'].annotate('%is' % args['Tbar'],
                                (args['tlim'][0], 1.005*args['Tbar_loc']),
                                 ha='left', fontsize=8,)
    # AX['axTraces'].set_xlim(args['tlim'])
    AX['axTraces'].set_ylim([-0.01, 1.01])

    return fig, AX, metadata


def get_pupil_center(index, data, metadata):
    coords = []
    for key in ['cx', 'cy']:
        coords.append(\
            data.nwbfile.processing['Pupil'].data_interfaces[key].data[index]/metadata['pix_to_mm'])
    return coords

def get_pupil_fit(index, data, metadata):
    coords = []
    for key in ['cx', 'cy', 'sx', 'sy']:
        coords.append(data.nwbfile.processing['Pupil'].data_interfaces[key].data[index]/metadata['pix_to_mm'])
    if 'angle' in data.nwbfile.processing['Pupil'].data_interfaces:
        coords.append(data.nwbfile.processing['Pupil'].data_interfaces['angle'].data[index])
    else:
        coords.append(0)
    return process.ellipse_coords(*coords, transpose=False)
    
def load_Imaging(metadata):
    metadata['raw_Imaging_folder'] = args['raw_Imaging_folder']

def fill_sheet_with_datafiles(nwbfile, args):
    """
    """

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)

    parser.add_argument("-v", "--verbose", 
                        help="increase output verbosity", 
                        action="store_true")

    parser.add_argument("--ROIs", default=[0,1,2,3,4],
                        nargs='*', type=int)
    parser.add_argument("--zoomROIs", default=[2,4],
                        nargs=2, type=int)

    parser.add_argument("--tlim", default=[10,100], 
                        nargs=2, type=float)

    parser.add_argument("--layout", help="show layout",
                        action="store_true")

    args = parser.parse_args()


    exec(string_params)

    if args.layout:

        fig, AX = layout(params)
        plt.show()

    if ('.nwb' in args.datafile) and os.path.isfile(args.datafile):

        params['zoomROIs'] = [21,9]
        params['ROIs'] = [21,16,9,1]

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


