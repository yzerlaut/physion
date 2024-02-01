# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from PIL import Image

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
    'nwbfile':"-",
    'raw_Behavior_folder':'',
    'raw_Imaging_folder':'',

    ############################################
    ###         VIEW OPTIONS     ###############
    ############################################
    'tlim':[10,100],

    # imaging
    'imaging_NL':3,
    'trace_quantity':'rawFluo',
    # ROIs zoom
    'zoomROIs':[0,1],

    # FaceCamera
    'FaceCameraLim':[0, 0, 10000, 10000],
    # RigCamera
    'RigCameraLim':[0, 0, 10000, 10000],

    ############################################
    ###      ANNOTATIONS         ###############
    ############################################
    'imaging_title':'',
    'Tbar':2, 'Tbar_loc':1.0,
    'with_screen_inset':True,

    ############################################
    ###       LAYOUT OPTIONS     ###############
    ############################################
    'ROIs':range(5),
    'fractions': {'running':0.13, 'running_start':0.,
                  'whisking':0.12, 'whisking_start':0.14,
                  'pupil':0.13, 'pupil_start':0.27,
                  'rois':0.35, 'rois_start':0.40,
                  'raster':0.24, 'raster_start':0.75,
                  'visual_stim':2., 'visual_stim_start':2.},
}
"""


def layout(args):

    height0, height1, width0, width1 = 0.65, 0.75, 0.79, 0.25
    
    AX = {}
    fig = plt.figure(figsize=(9,5))

    AX['axImaging'] = pt.inset(fig, 
                (width0, height0, 1-width0, 1-height0))
    if 'layout' in args and args['layout']:
        AX['axImaging'].imshow(np.zeros((2,2)), vmin=0, cmap=iMap)
    AX['axImaging'].axis('off')
    AX['axSetup'] = pt.inset(fig,
                (0, height1, .9*width1, .95*(1-height1)))
    keys = ['axRig', 'axFace', 'axScreen']
    titles = ['rig camera', 'face camera', 
              'screen\n(visual stimulation)']
    for i, key in enumerate(keys):
        AX[key] = pt.inset(fig,
              (width1+i*(width0-width1)/len(keys), height0+0.05,
              0.9*(width0-width1)/len(keys), 0.4*height0))
        AX[key].set_title(titles[i], fontsize=8)
        AX[key].axis('off')

    pt.annotate(AX['axImaging'], args['imaging_title'], (0.5,0.98),
                fontsize=7, va='top', ha='center', color='w')
    img = Image.open('../docs/exp-rig.png')
    AX['axSetup'].imshow(img)
    AX['axSetup'].axis('off')

    AX['axTraces'] = pt.inset(fig,(width1,0,1-width1,0.98*height0))

    keys = ['axWhisking', 'axPupil']+\
            ['axROI%i'%(n+1) for n in range(len(args['zoomROIs']))]
    titles = ['whisking\n(motion $\epsilon$)', 'pupil\n(ellipse fit)']+\
                ['cell #%i'%(n+1) for n in args['zoomROIs']]
    colors = ['tab:purple', 'red']+\
                ['tab:green' for n in range(len(args['zoomROIs']))]
    height1 *= 0.87
    for i, key in enumerate(keys):
        AX[key] = pt.inset(fig, (0.03, 0.07+i*height1/len(keys), 
                                 0.12, 0.9*height1/len(keys)))
        if 'layout' in args and args['layout']:
            AX[key].imshow(np.zeros((2,2)), vmin=0)
        AX[key].axis('equal')
        AX[key].axis('off')
        AX[key].annotate(titles[i],
                         (0.,0.5), va='center', ha='center',
                         rotation=90, fontsize=8, 
                         color=colors[i], xycoords='axes fraction')

    AX['axTime'] = pt.inset(fig, (0, 0, 0.05, 0.05))
    AX['axTime'].axis('off')

    if 'layout' in args and args['layout']:

        AX['axTime'].annotate(20*' '+'t=0.0s', (0,0),
                              xycoords='figure fraction', size=9)

    return fig, AX

        
def draw_figure(args, data):

    fig, AX = layout(args)

    metadata = dict(data.metadata)
    metadata['raw_Behavior_folder'] = args['raw_Behavior_folder']
    metadata['raw_Imaging_folder'] = args['raw_Imaging_folder']


    if 'ophys' in data.nwbfile.processing:

        # full image
        max_proj = getattr(getattr(data.nwbfile.processing['ophys'],
                           'data_interfaces')['Backgrounds_0'],
                           'images')['max_proj'][:]
        max_proj_scaled = np.power(max_proj/max_proj.max(),
                                   1/args['imaging_NL'])

        AX['imgImaging'] = AX['axImaging'].imshow(max_proj_scaled, 
                    vmin=0, vmax=1, cmap=iMap, origin='lower',
                    aspect='equal', interpolation='none')

        AX['axImaging'].annotate(' n=%i rois' % data.iscell.sum(),
                                  (0,0), color='w', fontsize=8,
                                  xycoords='axes fraction')

        # zoomed ROIs
        extents, max_projs = [], []
        for i, roi in enumerate(args['zoomROIs']):
            extents.append(find_roi_extent(data, roi, roi_zoom_factor=5))
            max_projs.append(\
                getattr(getattr(data.nwbfile.processing['ophys'],
                                'data_interfaces')['Backgrounds_0'],
                    'images')['max_proj'][:][extents[i][0]:extents[i][1],
                                             extents[i][2]:extents[i][3]])
            # max_proj_scaled1 = \
                # (max_proj-max_proj.min())/\
                # (max_proj.max()-max_proj.min())
            # max_proj_scaled1 = np.power(max_proj_scaled1,
                                          # 1/args['imaging_NL)

            AX['imgROI%i' % (i+1)] = \
                    AX['axROI%i' % (i+1)].imshow(max_projs[i],
                            vmin=0, vmax=max_projs[i].max(), 
                            cmap=iMap, extent=extents[i],
                            aspect='equal', interpolation='none', 
                            origin='lower')
            add_roi_ellipse(data, roi,
                            AX['axROI%i' % (i+1)],
                            size_factor=1.5, roi_lw=1)


    # setup drawing
    time = AX['axTime'].annotate(20*' '+'t=%.1fs\n' % args['tlim'][0],
                                (0,0), xycoords='figure fraction', size=8)

    # screen inset
    # AX['imgScreen'] = data.visual_stim.show_frame(0,
                                                  # ax=AX['axScreen'],
                                                  # return_img=True,
                                                  # label=None)

    # Calcium Imaging
    if metadata['raw_Imaging_folder']!='':
        
        Ly, Lx = getattr(getattr(data.nwbfile.processing['ophys'],
                         'data_interfaces')['Backgrounds_0'], 
                         'images')['meanImg'].shape
        Ca_data = BinaryFile(Ly=Ly, Lx=Lx,
                             read_filename=os.path.join(\
                                     metadata['raw_Imaging_folder'],
                                     'suite2p', 'plane0','data.bin'))
        i1, i2 = convert_times_to_indices(args['tlim'][0], args['tlim'][1],
                                          data.Fluorescence)

        imaging_scale = Ca_data.data[i1:i2,:,:].min(),\
                                    Ca_data.data[i1:i2,:,:].max()
        imaging_scales = []
        for n in range(len(args['zoomROIs'])):
            imaging_scales.append(\
                    (Ca_data.data[i1:i2,
                                 extents[n][0]:extents[n][1],
                                 extents[n][2]:extents[n][3]].min(),\
                    (Ca_data.data[i1:i2,
                                 extents[n][0]:extents[n][1],
                                 extents[n][2]:extents[n][3]].max())))


    # Face Camera
    if metadata['raw_Behavior_folder']!='':

        load_NIdaq(metadata)

        loadCameraData(metadata)

        # Rig Image
        img = np.load(metadata['raw_Rig_FILES'][0])
        AX['imgRig'] = AX['axRig'].imshow(imgRig_process(img,args),
                                    vmin=0, vmax=1, cmap='gray')

        # Face Image
        AX['imgFace'] = AX['axFace'].imshow(imgFace_process(img,args),
                                            vmin=0, vmax=1, cmap='gray')

        # pupil
        if 'pupil' in args['fractions']:
            x, y = np.meshgrid(np.arange(0,img.shape[0]),
                               np.arange(0,img.shape[1]), indexing='ij')
            pupil_cond = (y>=metadata['pupil_xmin']) &\
                         (y<=metadata['pupil_xmax']) &\
                         (x>=metadata['pupil_ymin']) &\
                         (x<=metadata['pupil_ymax'])
            pupil_shape = len(np.unique(x[pupil_cond])),\
                                    len(np.unique(y[pupil_cond]))
            AX['imgPupil'] = AX['axPupil'].imshow(\
                    img[pupil_cond].reshape(*pupil_shape), cmap='gray')
            pupil_fit = get_pupil_fit(0, data, metadata)
            AX['pupil_fit'], = AX['axPupil'].plot(pupil_fit[0],
                                                  pupil_fit[1],
                                '.', markersize=1, color='red')

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
            img1 = np.load(metadata['raw_Face_FILES'][1])

            new_img = (img1-img)[whisking_cond].reshape(*whisking_shape)
            AX['imgWhisking'] = AX['axWhisking'].imshow(new_img,
                                            vmin=-255/4, vmax=255+255/4,
                                            cmap=plt.cm.BrBG)


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
        AX['axTraces'].annotate('photodiode',
                    (-0.01, args['fraction']['photodiode_start']),
                                ha='right', va='bottom',
                                color='grey', fontsize=8,
                                xycoords='axes fraction')

    # locomotion
    if 'running' in args['fractions']:
        add_Locomotion(data, args['tlim'], AX['axTraces'], 
                    fig_fraction_start=args['fractions']['running_start'], 
                    fig_fraction=args['fractions']['running'], 
                    scale_side='right', subsampling=1,
                    name='')
        AX['axTraces'].annotate('running \nspeed \n ',
                            (-0.01, args['fractions']['running_start']),
                            ha='right', va='bottom',
                            color='#1f77b4', fontsize=8,
                            xycoords='axes fraction')

    # whisking 
    if 'whisking' in args['fractions']:
        add_FaceMotion(data, args['tlim'], AX['axTraces'], 
                fig_fraction_start=args['fractions']['whisking_start'], 
                fig_fraction=args['fractions']['whisking'], 
                scale_side='right', subsampling=1, name='')
        AX['axTraces'].annotate('whisking \n',
                        (-0.01, args['fractions']['whisking_start']),
                        ha='right', va='bottom',
                        color='purple', fontsize=8,
                        xycoords='axes fraction')

    # gaze 
    if 'gaze' in args['fractions']:
        add_GazeMovement(data, args['tlim'], AX['axTraces'], 
                fig_fraction_start=args['fractions']['gaze_start'], 
                fig_fraction=args['fractions']['gaze'], 
                scale_side='right', name='')
        AX['axTraces'].annotate('gaze \nmov. ',
                                (-0.01, args['fractions']['gaze_start']),
                                ha='right', va='bottom', 
                                color='orange', fontsize=8,
                                xycoords='axes fraction')

    # pupil 
    if 'pupil' in args['fractions']:
        add_Pupil(data, args['tlim'], AX['axTraces'], 
                    fig_fraction_start=args['fractions']['pupil_start'], 
                    fig_fraction=args['fractions']['pupil'], 
                    scale_side='right', subsampling=1, name='')
        AX['axTraces'].annotate('pupil \ndiam. ',
                                (-0.01, args['fractions']['pupil_start']),
                                ha='right', va='bottom',
                                color='red', fontsize=8,
                                xycoords='axes fraction')

    # rois 
    if 'ophys' in data.nwbfile.processing:
        data.build_rawFluo()
        add_CaImaging(data, args['tlim'], AX['axTraces'], 
                      subquantity=args['trace_quantity'],
                      roiIndices=args['ROIs'], 
                      fig_fraction_start=args['fractions']['rois_start'], 
                      fig_fraction=args['fractions']['rois'], 
                      scale_side='right',
                      name='', annotation_side='left')
        AX['axTraces'].annotate('fluorescence', (-0.1,
                    args['fractions']['pupil rois_start']+\
                            args['fractions']['rois']/2.),
                                ha='right', va='center', color='green',
                                rotation=90, xycoords='axes fraction')

        # raster 
        if 'raster' in args['fractions']:
            add_CaImagingRaster(data, args['tlim'], AX['axTraces'], 
                        subquantity='dFoF', 
                        normalization='per-line',
                        fig_fraction_start=args['fractions']['raster_start'], 
                        fig_fraction=args['fractions']['raster'], 
                        name='')

    if args['Tbar']>0:
        AX['axTraces'].plot(args['Tbar']*np.arange(2)+args['tlim'][0],
                            args['Tbar_loc']*np.ones(2), 'k-', lw=1)
        AX['axTraces'].annotate('%is' % args['Tbar'],
                                (args['tlim'][0], 1.005*args['Tbar_loc']),
                                 ha='left', fontsize=8,)

    AX['axTraces'].axis('off')
    AX['axTraces'].set_xlim(args['tlim'])
    AX['axTraces'].set_ylim([-0.01, 1.01])

    return fig, AX


def imgFace_process(img, args, exp=0.5,
                    bounds=[0.05, 0.75]):
    Img = (img-np.min(img))/(np.max(img)-np.min(img))
    Img = np.power(Img, exp) 
    # Img[Img<bounds[0]]=bounds[0]
    # Img[Img>bounds[1]]=bounds[1]
    # Img = 0.2+0.6*(Img-np.min(Img))/(np.max(Img)-np.min(Img))
    return Img[args['FaceCameraLim'][0]:args['FaceCameraLim'][2],\
               args['FaceCameraLim'][1]:args['FaceCameraLim'][3]] 

def imgRig_process(img, args):
    Img = (img-np.min(img))/(np.max(img)-np.min(img))
    return Img[args['RigCameraLim'][0]:args['RigCameraLim'][2],\
               args['RigCameraLim'][1]:args['RigCameraLim'][3]] 

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

    args = vars(parser.parse_args())
 
    exec(string_params)

    params['layout'] = args['layout']
    if args['layout']:

        fig, AX = layout(params)
        plt.show()

    elif '.nwb' in args['datafile']:

        data = physion.analysis.read_NWB.Data(args['datafile'],
                            with_visual_stim=bool(args['with_screen_inset']))
        fig, AX = draw_figure(params, data)    

        plt.show()

        # replace params in strings
        string_params=string_params.replace("[10,100]", str(args['tlim']))
        string_params=string_params.replace("range(5)", str(args['ROIs']))
        string_params=string_params.replace("[0,1]", str(args['zoomROIs']))
        string_params=string_params.replace('"-"', '"%s"'%args['datafile'])

        if 'y' in input('\n write ~/Desktop/snapshot.py file ? [N/y]\n   '):
            with open(os.path.join(os.path.expanduser('~'),
                                   'Desktop', 'snapshot.py'), 
                      'w') as f:
                f.write(string_params)

    elif os.path.isfile(args['datafile']):

        with open(args['datafile']) as f:
            string_params = f.read()
            exec(string_params)

        params['layout'] = args['layout']
        params['datafile'] = params['nwbfile']
        data = physion.analysis.read_NWB.Data(params['datafile'],
                            with_visual_stim=bool(params['with_screen_inset']))

        fig, AX = draw_figure(params, data) 

        plt.show()

    else:
        print('')
        print(' provide either a nwbfile or a snapshot.py file as argument')
        print('')


