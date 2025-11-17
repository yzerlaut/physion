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

default_params = """
{
    "                                                ":"",
    " ############################################## ":"",
    " ################# data location ############## ":"",
    " ############################################## ":"",
    "nwbfile":"~/DATA/physion_Demo-Datasets/PYR-WT/NWBs/2025_11_14-13-54-32.nwb",
    "raw_data_folder":"~/DATA/physion_Demo-Datasets/PYR-WT/processed/2025_11_14/13-54-32",
    "                                                ":"",
    " ############################################## ":"",
    " ############  data sample properties ######### ":"",
    " ############################################## ":"",
    "tlim":[20,120],
    "zoomROIs":[0,1],
    "                                                ":"",
    " ############################################## ":"",
    " #############  imaging properties ############ ":"",
    " ############################################## ":"",
    "imaging_temporal_filter":3.0,
    "imaging_spatial_filter":0.8,
    "imaging_NL":3,
    "imaging_clip":[0.3, 0.9],
    "trace_quantity":"rawFluo",
    "dFoF_smoothing":0.1,
    "zoomROIs_factor":[3.0,2.5],
    "                                                ":"",
    " ############################################## ":"",
    " ##########  Face-camera properties ########### ":"",
    " ############################################## ":"",
    "Face_Lim":[0, 0, 10000, 10000],
    "Face_clip":[0.3,1.0],
    "                                                ":"",
    " ############################################## ":"",
    " ##########  Rig-camera properties ############ ":"",
    " ############################################## ":"",
    "Rig_Lim":[100, 100, 470, 750],
    "Rig_NL":2,
    "                                                ":"",
    " ############################################## ":"",
    " ##########  annotation properties ############ ":"",
    " ############################################## ":"",
    "Tbar":2, 
    "Tbar_loc":1.0,
    "with_screen_inset":false,
    "                                                ":"",
    " ############################################## ":"",
    " ##########   layout properties  ############## ":"",
    " ############################################## ":"",
    "ROIs":[0,1,2,3,4,5],
    "fractions": {"running":0.1, "running_start":0.89,
                  "whisking":0.1, "whisking_start":0.78,
                  "gaze":0.08, "gaze_start":0.7,
                  "pupil":0.15, "pupil_start":0.55,
                  "rois":0.29, "rois_start":0.29,
                  "visual_stim":2, "visual_stim_start":2.0,
                  "raster":0.28, "raster_start":0.0},
    "                                                ":""
}
"""

string_params = """
params = {

    ############################################
    ###         VIEW OPTIONS     ###############
    ############################################
    'tlim':[20,80],
    'Ndiscret':100, # for movie only

}
"""


def layout(args, show_axes=False):
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

def draw_figure(args, params, data):

    fig, AX = layout(args)

    metadata = dict(data.metadata)
    # metadata['raw_Behavior_folder'] = args['raw_Behavior_folder']
    # metadata['raw_Imaging_folder'] = args['raw_Imaging_folder']


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


    # screen inset
    AX['imgScreen'] = data.visual_stim.show_frame(0,
                                                  ax=AX['axScreen'],
                                                  return_img=True,
                                                  label=None)

    # Face Camera
    if True:
        # metadata['raw_Behavior_folder']!='':

        loadCameraData(params, 
                       params['raw_data_folder'])

        # Rig Image
        if params['raw_Rig_FILES'] is not None:
            imgRig = np.load(\
                    params['raw_Rig_FILES'][0]).astype(float)
            params['norm_Rig'] = np.max(imgRig)
            AX['imgRig'] = AX['axRig'].imshow(\
                            show_img(imgRig, params, 'Rig'),
                vmin=params['Rig_clip'][0] if 'Rig_clip' in params else 0,
                vmax=params['Rig_clip'][1] if 'Rig_clip' in params else 1,
                            cmap='gray')

        # Face Image
        if params['raw_Face_FILES'] is not None:

            imgFace = np.load(\
                    params['raw_Face_FILES'][0]).astype(float)
            params['norm_Face'] = np.max(imgFace)
            AX['imgFace'] = AX['axFace'].imshow(\
                            show_img(imgFace, params, 'Face'),
            vmin=params['Face_clip'][0] if 'Face_clip' in params else 0,
            vmax=params['Face_clip'][1] if 'Face_clip' in params else 1,
                            cmap='gray')

            # pupil
            if 'pupil' in params['fractions']:
                x, y = np.meshgrid(np.arange(0,imgFace.shape[0]),
                                np.arange(0,imgFace.shape[1]), 
                                indexing='ij')
                pupil_cond = (y>=params['pupil_xmin']) &\
                            (y<=params['pupil_xmax']) &\
                            (x>=params['pupil_ymin']) &\
                            (x<=params['pupil_ymax'])
                pupil_shape = len(np.unique(x[pupil_cond])),\
                                        len(np.unique(y[pupil_cond]))
                AX['imgPupil'] = AX['axPupil'].imshow(\
                        imgFace[pupil_cond].reshape(*pupil_shape), cmap='gray')
                params['pupil_cond'] = pupil_cond
                params['pupil_shape'] = pupil_shape
                pupil_fit = get_pupil_fit(0, data, params)
                AX['pupil_fit'], = AX['axPupil'].plot(pupil_fit[0],
                                                    pupil_fit[1],
                                    '.', markersize=1, color='red')

            AX['pupil_center'] = None
            if 'gaze' in params['fractions']:
                pupil_center = get_pupil_center(0, data, params)
                AX['pupil_center'], = AX['axPupil'].plot(\
                                [pupil_center[0]], [pupil_center[1]], '.',
                                markersize=5, color='orange')

            # whisking
            if 'whisking' in params['fractions']:
                whisking_cond = (x>=params['whisking_ROI'][0]) &\
                (x<=(params['whisking_ROI'][0]+params['whisking_ROI'][2])) &\
                (y>=params['whisking_ROI'][1]) &\
                (y<=(params['whisking_ROI'][1]+params['whisking_ROI'][3]))
                whisking_shape = len(np.unique(x[whisking_cond])),\
                                        len(np.unique(y[whisking_cond]))
                params['whisking_cond'] = whisking_cond
                params['whisking_shape'] = whisking_shape

                img1 = np.load(\
                        params['raw_Face_FILES'][1]).astype(float)
                img = np.load(\
                        params['raw_Face_FILES'][0]).astype(float)
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
    AX['axTraces'].set_ylim([-0.01, 1.01])

    return fig, AX, params


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
    metadata['raw_Imaging_folder'] = params['raw_Imaging_folder']

def fill_sheet_with_datafiles(nwbfile, args):
    """
    """

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()

    parser.add_argument('-f', "--params_file", 
                        default = '',
                        type=str)

    # parser.add_argument("datafile", type=str)

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

        
        
    # exec(string_params)

    if args.layout:
        # just showing the current figure layout
        fig, AX = layout(args, show_axes=True)
        plt.show()

    elif args.params_file=='':
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

    elif '.json' in args.params_file and os.path.isfile(args.params_file):

        with open(args.params_file, 'r') as f:
            params = json.load(f)

        data = physion.analysis.read_NWB.Data(os.path.expanduser(params['nwbfile']),
                                              with_visual_stim=True)
        
        # print('tlim: %s' % data.tlim)

        fig, AX, metadata = draw_figure(args, params, data)    

        plt.show()
        # root_path = os.path.dirname(args.datafile)
        # subfolder = os.path.basename(\
        #         args.datafile).replace('.nwb','')[-8:]

    else:
        print("""
        invalid input file
            """)
        
        # json.load()


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
