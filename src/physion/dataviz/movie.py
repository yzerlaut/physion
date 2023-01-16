# general modules
import pynwb, os, sys, pathlib, itertools
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as animation
from PIL import Image

from physion.utils.files import get_files_with_extension, list_dayfolder, get_TSeries_folders
from physion.assembling.tools import StartTime_to_day_seconds, load_FaceCamera_data
from physion.assembling.IO.binary import BinaryFile
import  physion.utils.plot_tools as pt
from physion.pupil import roi, process

def draw_figure(args, data,
                top_row_bottom=0.75,
                top_row_space=0.08,
                top_row_height=0.2,
                Ndiscret=100):

    metadata = dict(data.metadata)

    metadata['raw_vis_folder'] = args.raw_vis_folder
    metadata['raw_imaging_folder'] = args.raw_imaging_folder

    times = np.linspace(args.tlim[0], args.tlim[1], args.Ndiscret)


    fractions = {'photodiode':0.09, 'photodiode_start':0,
                 'running':0.13, 'running_start':0.1,
                 'whisking':0.15, 'whisking_start':0.25,
                 'gaze':0.1, 'gaze_start':0.35,
                 'pupil':0.13, 'pupil_start':0.45,
                 'rois':0.2, 'rois_start':0.6,
                 'raster':0.2, 'raster_start':0.8}

    AX = {'time_plot_ax':None}
    fig, AX['time_plot_ax'] = plt.subplots(1, figsize=(7,4))
    plt.subplots_adjust(bottom=0.01, right=0.97, left=0.3, top=0.7)

    width = (1.-4*top_row_space)/4.
    AX['setup_ax'] = pt.inset(fig,
            (top_row_space/2.+0*(width+top_row_space),
            top_row_bottom, width, top_row_height))
    AX['screen_ax'] = pt.inset(fig,
            (top_row_space/2.+1*(width+.5*top_row_space),
            top_row_bottom, 1.3*width, top_row_height))
    AX['camera_ax'] = pt.inset(fig,
            (top_row_space/2.+2*(width+top_row_space),
            top_row_bottom, width, top_row_height))

    if 'ophys' in data.nwbfile.processing:

        # full image
        max_proj = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'][:]
        max_proj_scaled = (max_proj-max_proj.min())/(max_proj.max()-max_proj.min())
        max_proj_scaled = np.power(max_proj_scaled, 1/args.imaging_NL)

        AX['imaging_ax'] = ge.inset(fig, (top_row_space/2.+3*(width+top_row_space), top_row_bottom-.04, width, top_row_height+0.08))
        ge.annotate(AX['imaging_ax'], 'imaging', (-0.05,0.5), ha='right', va='center', rotation=90)
        AX['imaging_img'] = AX['imaging_ax'].imshow(max_proj_scaled, vmin=0, vmax=1, 
                cmap=ge.get_linear_colormap('k','lightgreen'), 
                aspect='equal', interpolation='none', origin='lower')
        ge.annotate(AX['imaging_ax'], ' n=%i rois' % data.iscell.sum(), (0,0), color='w', size='xxx-small')

        # ROI 1 
        AX['ROI1_ax'] = ge.inset(fig, [0.04,0.6,0.11,0.13]) 
        extent1 = data.find_roi_extent(args.ROIs[0], roi_zoom_factor=4)
        max_proj = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'][:][extent1[0]:extent1[1], extent1[2]:extent1[3]]
        max_proj_scaled1 = (max_proj-max_proj.min())/(max_proj.max()-max_proj.min())
        max_proj_scaled1 = np.power(max_proj_scaled1, 1/args.imaging_NL)

        AX['ROI1_img'] = AX['ROI1_ax'].imshow(max_proj_scaled1, vmin=0, vmax=1, 
                cmap=ge.get_linear_colormap('k','lightgreen'), extent=extent1,
                aspect='equal', interpolation='none', origin='lower')
        ge.annotate(AX['ROI1_ax'], ' roi #%i' % (args.ROIs[0]+1), (0,0), color='w', size='xxx-small')
        data.add_roi_ellipse(args.ROIs[0], AX['ROI1_ax'], size_factor=1.5, roi_lw=1)

        # ROI 2 
        AX['ROI2_ax'] = ge.inset(fig, [0.04,0.45,0.11,0.13]) 
        extent2 = data.find_roi_extent(args.ROIs[1], roi_zoom_factor=4)
        max_proj = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'][:][extent2[0]:extent2[1], extent2[2]:extent2[3]]
        max_proj_scaled2 = (max_proj-max_proj.min())/(max_proj.max()-max_proj.min())
        max_proj_scaled2 = np.power(max_proj_scaled2, 1/args.imaging_NL)

        AX['ROI2_img'] = AX['ROI2_ax'].imshow(max_proj_scaled2, vmin=0, vmax=1, 
                cmap=ge.get_linear_colormap('k','lightgreen'), extent=extent2, 
                aspect='equal', interpolation='none', origin='lower')
        ge.annotate(AX['ROI2_ax'], ' roi #%i' % (args.ROIs[1]+1), (0,0), color='w', size='xxx-small')
        data.add_roi_ellipse(args.ROIs[1], AX['ROI2_ax'], size_factor=1.5, roi_lw=1)

    AX['whisking_ax'] = pt.inset(fig, [0.04,0.15,0.11,0.11]) 
    # ge.annotate(AX['whisking_ax'], '$F_{(t+dt)}$-$F_{(t)}$',
            # (0,0.5), ha='right', va='center', rotation=90, size='xxx-small')
    # ge.annotate(AX['whisking_ax'], 'motion frames', (0.5,0), ha='center', va='top', size='xxx-small')
    AX['pupil_ax'] = pt.inset(fig, [0.04,0.28,0.11,0.13]) 
    AX['time_ax'] = pt.inset(fig, [0.02,0.05,0.08,0.05]) 

    t0 = times[0]

    # setup drawing
    img = Image.open('../doc/exp-rig.png')
    AX['setup_ax'].imshow(img)
    AX['setup_ax'].axis('off')
    time = AX['time_ax'].annotate('t=%.1fs' % times[0], (0,0), xycoords='axes fraction', size=12)

    # screen inset
    AX['screen_img'] = data.visual_stim.show_frame(0, ax=AX['screen_ax'],
                                                   return_img=True,
                                                   label=None)

    """
    # Calcium Imaging
    if metadata['raw_imaging_folder']!='':
        
        Ly, Lx = data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'].shape
        Ca_data = BinaryFile(Ly=Ly, Lx=Lx,
                             read_filename=os.path.join(metadata['raw_imaging_folder'],
                                        'suite2p', 'plane0','data.bin'))
        i1, i2 = dv_tools.convert_times_to_indices(times[0], times[1], data.Fluorescence)

        imaging_scale = Ca_data.data[i1:i2,:,:].min(), Ca_data.data[i1:i2,:,:].max()

        imaging_scale1 = Ca_data.data[i1:i2, extent1[0]:extent1[1], extent1[2]:extent1[3]].min(),\
               Ca_data.data[i1:i2, extent1[0]:extent1[1], extent1[2]:extent1[3]].max() 

        imaging_scale2 = Ca_data.data[i1:i2, extent2[0]:extent2[1], extent2[2]:extent2[3]].min(),\
               Ca_data.data[i1:i2, extent2[0]:extent2[1], extent2[2]:extent2[3]].max() 

    else:
        Ca_data = None


    # Face Camera
    if metadata['raw_vis_folder']!='':
        load_NIdaq(metadata)
        load_faceCamera(metadata)

        img = np.load(metadata['raw_vis_FILES'][0])
        AX['camera_img'] = AX['camera_ax'].imshow(img, cmap='gray')
        # pupil
        x, y = np.meshgrid(np.arange(0,img.shape[0]), np.arange(0,img.shape[1]), indexing='ij')
        pupil_cond = (x>=metadata['pupil_xmin']) & (x<=metadata['pupil_xmax']) & (y>=metadata['pupil_ymin']) & (y<=metadata['pupil_ymax'])
        pupil_shape = metadata['pupil_xmax']-metadata['pupil_xmin']+1, metadata['pupil_ymax']-metadata['pupil_ymin']+1
        AX['pupil_img'] = AX['pupil_ax'].imshow(img[pupil_cond].reshape(*pupil_shape), cmap='gray')
        pupil_fit = get_pupil_fit(0, data, metadata)
        AX['pupil_fit'], = AX['pupil_ax'].plot(pupil_fit[0], pupil_fit[1], 'o', markersize=3, color=ge.red)
        pupil_center = get_pupil_center(0, data, metadata)
        AX['pupil_center'], = AX['pupil_ax'].plot([pupil_center[1]], [pupil_center[0]], 'o', markersize=6, color=ge.orange)

        # whisking
        whisking_cond = (x>=metadata['whisking_ROI'][0]) & (x<=(metadata['whisking_ROI'][0]+metadata['whisking_ROI'][2])) &\
                (y>=metadata['whisking_ROI'][1]) & (y<=(metadata['whisking_ROI'][1]+metadata['whisking_ROI'][3]))
        whisking_shape = len(np.unique(x[whisking_cond])), len(np.unique(y[whisking_cond]))
        img1 = np.load(metadata['raw_vis_FILES'][1])
        AX['whisking_img'] = AX['whisking_ax'].imshow((img1-img)[whisking_cond].reshape(*whisking_shape), cmap='gray')

    AX['setup_ax'].axis('off')
    
    # time cursor
    cursor, = AX['time_plot_ax'].plot(np.ones(2)*times[0], np.arange(2), 'k-')#color=ge.grey, lw=3, alpha=.3) 

    #   ----  filling time plot

    # photodiode and visual stim
    data.add_VisualStim(args.tlim, AX['time_plot_ax'], 
                        fig_fraction=2.,
                        with_screen_inset=False,
                        name='')
    data.add_Photodiode(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['photodiode_start'], 
                        fig_fraction=fractions['photodiode'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'photodiode', (-0.1, fractions['photodiode_start']), ha='center', va='bottom', color=ge.grey, size='small')


    # locomotion
    data.add_Locomotion(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['running_start'], 
                        fig_fraction=fractions['running'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'running-speed', (-0.1, fractions['running_start']), ha='center', va='bottom', color=ge.blue, size='small')

    # whisking 
    data.add_FaceMotion(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['whisking_start'], 
                        fig_fraction=fractions['whisking'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'whisking  ', (-0.01, fractions['whisking_start']), ha='right', va='bottom', color=ge.purple, size='small')

    # gaze 
    data.add_GazeMovement(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['gaze_start'], 
                        fig_fraction=fractions['gaze'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'gaze \nmov. ', (-0.01, fractions['gaze_start']), ha='right', va='bottom', color=ge.orange, size='small')

    # pupil 
    data.add_Pupil(args.tlim, AX['time_plot_ax'], 
                        fig_fraction_start=fractions['pupil_start'], 
                        fig_fraction=fractions['pupil'], 
                        name='')
    ge.annotate(AX['time_plot_ax'], 'pupil \ndiam. ', (-0.01, fractions['pupil_start']), ha='right', va='bottom', color=ge.red, size='small')

    # rois 
    data.add_CaImaging(args.tlim, AX['time_plot_ax'], 
                       roiIndices=args.ROIs, 
                       fig_fraction_start=fractions['rois_start'], 
                       fig_fraction=fractions['rois'], 
                       name='', annotation_side='left')
    ge.annotate(AX['time_plot_ax'], 'fluorescence', (-0.1, fractions['raster_start']), ha='right', va='center', color=ge.green, rotation=90)

    # raster 
    data.add_CaImagingRaster(args.tlim, AX['time_plot_ax'], 
                             subquantity='dFoF', normalization='per-line',
                             fig_fraction_start=fractions['raster_start'], 
                             fig_fraction=fractions['raster'], 
                             name='')
    

    if args.Tbar>0:
        AX['time_plot_ax'].plot(args.Tbar*np.arange(2)+times[0], args.Tbar_loc*np.ones(2), 'k-')
        ge.annotate(AX['time_plot_ax'], '%is' % args.Tbar, (0,args.Tbar_loc+0.01))
    ge.set_plot(AX['time_plot_ax'], [], xlim=[times[0], times[-1]], ylim=[-0.01, 1.01])

    for i, label in enumerate(['screen', 'camera']):
        ge.set_plot(AX['%s_ax'%label], [], title=label)
    for i, label in enumerate(['imaging', 'pupil', 'whisking', 'ROI1', 'ROI2', 'time']):
        ge.set_plot(AX['%s_ax'%label], [], title=' ')

    def update(i=0):
        # camera
        camera_index = np.argmin((metadata['raw_vis_times']-times[i])**2)
        img = np.load(metadata['raw_vis_FILES'][camera_index])
        AX['camera_img'].set_array(img)
        # pupil
        AX['pupil_img'].set_array(img[pupil_cond].reshape(*pupil_shape))
        pupil_fit = get_pupil_fit(camera_index, data, metadata)
        AX['pupil_fit'].set_data(pupil_fit[1], pupil_fit[0])
        pupil_center = get_pupil_center(camera_index, data, metadata)
        AX['pupil_center'].set_data([pupil_center[1]], [pupil_center[0]])
        # whisking
        img1 = np.load(metadata['raw_vis_FILES'][camera_index+1])
        AX['whisking_img'].set_array((img1-img)[whisking_cond].reshape(*whisking_shape))

        # imaging
        if (i in [0,len(times)-1]) or (Ca_data is None):
            AX['imaging_img'].set_array(max_proj_scaled)
            AX['ROI1_img'].set_array(max_proj_scaled1)
            AX['ROI2_img'].set_array(max_proj_scaled2)
        else:
            im_index = dv_tools.convert_time_to_index(times[i], data.Fluorescence)
            img = Ca_data.data[im_index,:,:].astype(np.uint16)
            img = (img-imaging_scale[0])/(imaging_scale[1]-imaging_scale[0])
            img = np.power(img, 1/args.imaging_NL)
            AX['imaging_img'].set_array(img)

            img1 = Ca_data.data[im_index, extent1[0]:extent1[1], extent1[2]:extent1[3]]
            img1 = (img1-imaging_scale1[0])/(imaging_scale1[1]-imaging_scale1[0])
            img1 = np.power(img1, 1/args.imaging_NL)
            AX['ROI1_img'].set_array(img1)

            img2 = Ca_data.data[im_index, extent2[0]:extent2[1], extent2[2]:extent2[3]]
            img2 = (img2-imaging_scale2[0])/(imaging_scale2[1]-imaging_scale2[0])
            img2 = np.power(img2, 1/args.imaging_NL)
            AX['ROI2_img'].set_array(img2)
            

        # visual stim
        iEp = data.find_episode_from_time(times[i])
        if iEp==-1:
            AX['screen_img'].set_array(data.visual_stim.x*0+0.5)
        else:
            tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
            data.visual_stim.update_frame(iEp, AX['screen_img'],
                                          time_from_episode_start=times[i]-tEp)
        cursor.set_data(np.ones(2)*times[i], np.arange(2))
        # time
        time.set_text('t=%.1fs' % times[i])
        
        return [cursor, time, AX['screen_img'], AX['camera_img'], AX['pupil_img'],
                AX['whisking_img'], AX['pupil_fit'], AX['pupil_center'],
                AX['imaging_img'], AX['ROI1_img'], AX['ROI2_img']]
        
    ani = animation.FuncAnimation(fig, 
                                  update,
                                  np.arange(len(times)),
                                  init_func=update,
                                  interval=100,
                                  blit=True)

    return fig, AX, ani
    """
    pt.plt.show()

def get_pupil_center(index, data, metadata):
    coords = []
    for key in ['cx', 'cy']:
        coords.append(data.nwbfile.processing['Pupil'].data_interfaces[key].data[index]/metadata['pix_to_mm'])
    return coords

def get_pupil_fit(index, data, metadata):
    coords = []
    for key in ['cx', 'cy', 'sx', 'sy']:
        coords.append(data.nwbfile.processing['Pupil'].data_interfaces[key].data[index]/metadata['pix_to_mm'])
    if 'angle' in data.nwbfile.processing['Pupil'].data_interfaces:
        coords.append(data.nwbfile.processing['Pupil'].data_interfaces['angle'].data[index])
    else:
        coords.append(0)
    return process.ellipse_coords(*coords)
    
def load_faceCamera(metadata):
    imgfolder = os.path.join(metadata['raw_vis_folder'], 'FaceCamera-imgs')
    times, FILES, nframes, Lx, Ly = load_FaceCamera_data(imgfolder, t0=metadata['NIdaq_Tstart'], verbose=True)
    metadata['raw_vis_times'] = times 
    metadata['raw_vis_FILES'] = [os.path.join(imgfolder, f) for f in FILES]
    dataP = np.load(os.path.join(metadata['raw_vis_folder'], 'pupil.npy'),
                                 allow_pickle=True).item()
    for key in dataP:
        metadata['pupil_'+key] = dataP[key]
    dataW = np.load(os.path.join(metadata['raw_vis_folder'], 'facemotion.npy'),
                                 allow_pickle=True).item()
    for key in dataW:
        metadata['whisking_'+key] = dataW[key]

    if 'FaceCamera-1cm-in-pix' in metadata:
        metadata['pix_to_mm'] = 10./float(metadata['FaceCamera-1cm-in-pix']) # IN MILLIMETERS FROM HERE
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

    parser.add_argument("-rvf", '--raw_vis_folder', type=str, default='')
    parser.add_argument("-rif", '--raw_imaging_folder', type=str, default='')
    
    parser.add_argument("--tlim", type=float, nargs='*', default=[1, 10], help='')
    parser.add_argument("--Tbar", type=int, default=0)
    parser.add_argument("--Tbar_loc", type=float, default=0.1, help='y-loc of Tbar in [0,1]')

    parser.add_argument('-rois', "--ROIs", type=int, default=[0,1], nargs='*')
    parser.add_argument('-n', "--Ndiscret", type=int, default=10)
    parser.add_argument('-q', "--quantity", type=str, default='dFoF')

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("-e", "--export", help="export to mp4", action="store_true")
    # video properties
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--duration", type=float, default=0, help='video duration')
    parser.add_argument("--dpi", type=int, default=100, help='video duration')

    parser.add_argument("--imaging_NL", type=int, default=3, help='1/exponent for image transform')

    args = parser.parse_args()

    if args.duration>0:
        args.Ndiscret = int(args.duration*args.fps)

    data = physion.analysis.read_NWB.Data(args.datafile, with_visual_stim=True)
    print('\n', data.nwbfile.processing['ophys'].description, '\n')

    fig, AX, ani = draw_figure(args, data)    
    if args.export:
        print('writing video [...]')
        writer = animation.writers['ffmpeg'](fps=args.fps)
        ani.save('demo.mp4',writer=writer,dpi=args.dpi)# fig, ax = ge.twoD_plot(np.arange(50), np.arange(30), np.random.randn(50, 30))
    else:
        ge.show()




