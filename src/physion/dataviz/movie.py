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


iMap = pt.get_linear_colormap('k','lightgreen')

fractions = {'photodiode':0.06, 'photodiode_start':0,
             'running':0.13, 'running_start':0.07,
             'whisking':0.12, 'whisking_start':0.2,
             'gaze':0.1, 'gaze_start':0.32,
             'pupil':0.13, 'pupil_start':0.4,
             'rois':0.27, 'rois_start':0.53,
             'raster':0.2, 'raster_start':0.8}

def layout(args,
           top_row_bottom=0.75,
           top_row_space=0.08,
           top_row_height=0.2):

    
    AX = {}
    fig = plt.figure(figsize=(9,5))

    height0, height1, width0, width1 = 0.63, 0.75, 0.79, 0.25
    AX['axImaging'] = pt.inset(fig, 
                (width0, height0, 1-width0, 1-height0))
    if args.layout:
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

    pt.annotate(AX['axImaging'], args.imaging_title, (0.5,0.98),
                fontsize=7, va='top', ha='center', color='w')
    img = Image.open('../docs/exp-rig.png')
    AX['axSetup'].imshow(img)
    AX['axSetup'].axis('off')

    AX['axTraces'] = pt.inset(fig,(width1,0,1-width1,0.98*height0))

    keys = ['axWhisking', 'axPupil', 'axROI1', 'axROI2', 'axROI3']
    titles = ['whisking', 'pupil',
              'neuron #1', 'neuron #2', 'neuron #3']
    for i, key in enumerate(keys):
        AX[key] = pt.inset(fig, (0.03,
                                 i*height1/len(keys), 
                                 0.15,
                                 0.9*height1/len(keys)))
        if args.layout:
            AX[key].imshow(np.zeros((2,2)), vmin=0)
        AX[key].axis('equal')
        pt.annotate(AX[key], titles[i], (0.5,1), 
                    ha='center', va='top', color='w')
        AX[key].axis('off')
    AX['axTime'] = pt.inset(fig, (0, 0, 0.05, 0.05))

    return fig, AX

        
def draw_figure(args, data,
                Ndiscret=100):

    fig, AX = layout(args)

    metadata = dict(data.metadata)

    metadata['raw_vis_folder'] = args.raw_vis_folder
    metadata['raw_imaging_folder'] = args.raw_imaging_folder

    times = np.linspace(args.tlim[0], args.tlim[1], args.Ndiscret)


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
                              (0.,0.5), va='center', ha='right',
                              fontsize=7, xycoords='axes fraction')

    AX['axPupil'].annotate('ellipse\nfitting',
                              (0.,0.5), va='center', ha='right',
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

        for n in range(3):
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
        AX['imgRig'] = AX['axRig'].imshow(imgRig_process(img),
                                    vmin=0, vmax=1, cmap='gray')

        # Pupil Image
        img = np.load(metadata['raw_Face_FILES'][0])
        AX['imgFace'] = AX['axFace'].imshow(imgFace_process(img),
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
                       fig_fraction=2., with_screen_inset=False,
                       name='')
        add_Photodiode(data, args.tlim, AX['axTraces'], 
                       fig_fraction_start=fractions['photodiode_start'], 
                       fig_fraction=fractions['photodiode'], name='')
        AX['axTraces'].annotate('photodiode', (-0.01, fractions['photodiode_start']),
                ha='right', va='bottom', color='grey', fontsize=8, xycoords='axes fraction')


    # locomotion
    add_Locomotion(data, args.tlim, AX['axTraces'], 
                        fig_fraction_start=fractions['running_start'], 
                        fig_fraction=fractions['running'], 
                        scale_side='right',
                        name='')
    AX['axTraces'].annotate('locomotion \nspeed \n ', (-0.01, fractions['running_start']),
            ha='right', va='bottom', color='#1f77b4', fontsize=8, xycoords='axes fraction')

    # whisking 
    add_FaceMotion(data, args.tlim, AX['axTraces'], 
                   fig_fraction_start=fractions['whisking_start'], 
                   fig_fraction=fractions['whisking'], 
                   scale_side='right',
                   name='')
    AX['axTraces'].annotate('whisking \n', (-0.01, fractions['whisking_start']),
            ha='right', va='bottom', color='purple', fontsize=8, xycoords='axes fraction')

    # gaze 
    add_GazeMovement(data, args.tlim, AX['axTraces'], 
                        fig_fraction_start=fractions['gaze_start'], 
                        fig_fraction=fractions['gaze'], 
                  scale_side='right',
                        name='')
    AX['axTraces'].annotate('gaze \nmov. ', (-0.01, fractions['gaze_start']),
            ha='right', va='bottom', color='orange', fontsize=8, xycoords='axes fraction')

    # pupil 
    add_Pupil(data, args.tlim, AX['axTraces'], 
                        fig_fraction_start=fractions['pupil_start'], 
                        fig_fraction=fractions['pupil'], 
                        scale_side='right',
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
        # AX['dFoFscale_ax'] = pt.inset(fig,
                        # [0.2, top_row_bottom*0.95*(fractions['raster_start']+.2*fractions['raster']),
                         # 0.01, top_row_bottom*0.95*0.6*fractions['raster']], facecolor=None)
        add_CaImagingRaster(data, args.tlim, AX['axTraces'], 
                    subquantity='dFoF', 
                    normalization='per-line',
                    fig_fraction_start=fractions['raster_start'], 
                    fig_fraction=fractions['raster'], 
                    # axb = AX['dFoFscale_ax'],
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
            print(True)
            camera_index = np.argmin((metadata['raw_Rig_times']-times[i])**2)
            img = np.load(metadata['raw_Rig_FILES'][camera_index])
            AX['imgRig'].set_array(img)

        if 'raw_Face_times' in metadata:
            # Face camera
            camera_index = np.argmin((metadata['raw_Face_times']-times[i])**2)
            img = np.load(metadata['raw_Face_FILES'][camera_index])
            AX['imgFace'].set_array(img)
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
            for n in range(3):
                AX['imgROI%i' % (n+1)].set_array(max_projs[n])
        else:
            im_index = dv_tools.convert_time_to_index(times[i], data.Fluorescence)
            img = Ca_data.data[im_index,:,:].astype(np.uint16)
            img = np.power(img/np.max(max_proj), 1/args.imaging_NL)
            AX['imgImaging'].set_array(img)

            for n in range(3):
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

        return fig, AX, ani
    else:
        return fig, AX, None 


def imgFace_process(img, exp=0.1,
                    bounds=[0.05, 0.75]):
    Img = (img-np.min(img))/(np.max(img)-np.min(img))
    # Img = np.power(Img, exp) 
    Img[Img<bounds[0]]=bounds[0]
    Img[Img>bounds[1]]=bounds[1]
    Img = 0.2+0.6*(Img-np.min(Img))/(np.max(Img)-np.min(Img))
    return Img

def imgRig_process(img):
    Img = (img-np.min(img))/(np.max(img)-np.min(img))
    return Img[10:,:-150] 

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
    
def loadCameraData(metadata):
    # FaceCamera
    imgfolder = os.path.join(metadata['raw_vis_folder'],
                             'FaceCamera-imgs')
    times, FILES, nframes, Lx, Ly =\
            load_FaceCamera_data(imgfolder, 
                                 t0=metadata['NIdaq_Tstart'], 
                                 verbose=True)
    metadata['raw_Face_times'] = times 
    metadata['raw_Face_FILES'] = \
            [os.path.join(imgfolder, f) for f in FILES]
    # RigCamera
    imgfolder = os.path.join(metadata['raw_vis_folder'],
                             'RigCamera-imgs')
    times, FILES, nframes, Lx, Ly =\
            load_FaceCamera_data(imgfolder, 
                                 t0=metadata['NIdaq_Tstart'], 
                                 verbose=True)
    metadata['raw_Rig_times'] = times 
    metadata['raw_Rig_FILES'] = \
            [os.path.join(imgfolder, f) for f in FILES]
    dataP = np.load(os.path.join(metadata['raw_vis_folder'], 
                                 'pupil.npy'),
                                 allow_pickle=True).item()
    for key in dataP:
        metadata['pupil_'+key] = dataP[key]
    dataW = np.load(os.path.join(metadata['raw_vis_folder'],
                                 'facemotion.npy'),
                                  allow_pickle=True).item()
    for key in dataW:
        metadata['whisking_'+key] = dataW[key]

    if 'FaceCamera-1cm-in-pix' in metadata:
        metadata['pix_to_mm'] = \
                10./float(metadata['FaceCamera-1cm-in-pix']) # IN MILLIMETERS FROM HERE
    else:
        metadata['pix_to_mm'] = 1
        

def load_NIdaq(metadata):
    metadata['NIdaq_Tstart'] = np.load(os.path.join(metadata['raw_vis_folder'], 'NIdaq.start.npy'))[0]

def load_Imaging(metadata):
    metadata['raw_imaging_folder'] = args.raw_imaging_folder

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)

    parser.add_argument("-rvf", '--raw_vis_folder', 
                        type=str, default='')
    # IMAGING props
    parser.add_argument("-rif", '--raw_imaging_folder', 
                        type=str, default='')
    parser.add_argument('--imaging_title', type=str, 
                        default='GCamp6s fluorescence')
    
    parser.add_argument("--tlim", type=float, nargs='*', 
                        default=[10, 100], help='')
    parser.add_argument("--Tbar", type=int, default=0)
    parser.add_argument("--Tbar_loc", type=float, 
                        default=1.005, help='y-loc of Tbar in [0,1]')

    parser.add_argument("--no_visual", 
                        help="remove visual stimulation", 
                        action="store_true")

    parser.add_argument('-rois', "--ROIs", type=int, 
                        default=[0,1,2], nargs=3)
    parser.add_argument('-n', "--Ndiscret", type=int, default=10)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--layout", help="show layout",
                        action="store_true")
    parser.add_argument("-e", "--export", help="export to mp4", action="store_true")
    parser.add_argument("--snapshot", help="export to mp4", action="store_true")
    parser.add_argument('-o', "--output", type=str, default='demo.mp4')
    # video properties
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=0, help='video duration')
    parser.add_argument("--dpi", type=int, default=100, help='video duration')

    parser.add_argument("--imaging_NL", type=int, default=3, help='1/exponent for image transform')

    args = parser.parse_args()

    if args.duration>0:
        args.Ndiscret = int(args.duration*args.fps)

    # print('\n', data.nwbfile.processing['ophys'].description, '\n')


    if args.layout:

        fig, AX = layout(args)
        plt.show()

    else:
        data = physion.analysis.read_NWB.Data(args.datafile,
                                              with_visual_stim=True)
        fig, AX, ani = draw_figure(args, data)    
        print(ani)

        if args.export:
            print('writing video [...]')
            writer = animation.writers['ffmpeg'](fps=args.fps)
            ani.save(args.output, writer=writer, dpi=args.dpi)

        else:
            plt.show()




