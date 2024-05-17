# all the modules from "snapshot.py"
from physion.dataviz.snapshot import *
# + all the layout is from "snapshot.py"

import matplotlib.animation as animation

def draw_movie(args, data,
               Ndiscret=100):

    fig, AX, metadata = draw_figure(args, data)

    times = np.linspace(args['tlim'][0], args['tlim'][1], 
                        args['Ndiscret'])

    # Calcium Imaging
    if metadata['raw_Imaging_folder']!='':
        
        Ly, Lx = getattr(getattr(data.nwbfile.processing['ophys'],
                         'data_interfaces')['Backgrounds_0'], 
                         'images')['meanImg'].shape
        Ca_data = BinaryFile(Ly=Ly, Lx=Lx,
                             read_filename=os.path.join(\
                                     metadata['raw_Imaging_folder'],
                                     'suite2p', 'plane0','data.bin'))
    else:
        Ca_data = None

    def update(i=0):

        if 'raw_Rig_times' in metadata:
            # Rig camera
            camera_index = np.argmin((metadata['raw_Rig_times']\
                    -times[i])**2)
            img = np.load(\
            metadata['raw_Rig_FILES'][camera_index]).astype(float)
            AX['imgRig'].set_array(show_img(img, args, 'Rig'))

        if 'raw_Face_times' in metadata:
            # Face camera
            camera_index = np.min([np.argmin(\
                    (metadata['raw_Face_times']-times[i])**2),
                            len(metadata['raw_Face_times'])-2])
            img = np.load(metadata['raw_Face_FILES'][\
                                    camera_index]).astype(float)
            AX['imgFace'].set_array(show_img(img, args, 'Face'))
            # pupil
            AX['imgPupil'].set_array(\
                    img[metadata['pupil_cond']].reshape(\
                            *metadata['pupil_shape']))
            pupil_fit = get_pupil_fit(camera_index, 
                                      data, metadata)
            AX['pupil_fit'].set_data(pupil_fit[0], pupil_fit[1])
            pupil_center = get_pupil_center(camera_index, data, metadata)
            AX['pupil_center'].set_data([pupil_center[0]],
                                        [pupil_center[1]])
            # whisking
            img1 = np.load(metadata['raw_Face_FILES'][\
                                camera_index+1]).astype(float)
            img = np.load(metadata['raw_Face_FILES'][\
                                camera_index]).astype(float)
            new_img = (img1-img)[metadata['whisking_cond']\
                            ].reshape(*metadata['whisking_shape'])
            AX['imgWhisking'].set_array(new_img)

        # imaging
        if (i in [0,len(times)-1]) or (Ca_data is None):
            pass
        else:
            im_index = dv_tools.convert_time_to_index(times[i],
                                                      data.Fluorescence)
            dS = int(3*args['imaging_spatial_filter'])
            img = Ca_data.data[im_index-dS:im_index+dS+1,
                               :,:].astype(np.uint16).mean(axis=0)
            img = scipy.ndimage.gaussian_filter1d(img,
                                    args['imaging_spatial_filter'])
            img = scipy.ndimage.gaussian_filter(img, 
                                args['imaging_spatial_filter'])
            AX['imgImaging'].set_array(\
                    show_img(img, args, 'imaging'))

            for n, roi in enumerate(args['zoomROIs']):
                extent = args['ROI%i_extent'%n]
                img_ROI = img[extent[0]:extent[1], extent[2]:extent[3]] 
                AX['imgROI%i' % (n+1)].set_array(\
                            show_img(img_ROI, args, 'ROI%i'%n))


        # visual stim
        iEp = data.find_episode_from_time(times[i])
        if iEp==-1:
            AX['imgScreen'].set_array(data.visual_stim.x*0+0.5)
        else:
            tEp = data.nwbfile.stimulus['time_start_realigned'].data[iEp]
            data.visual_stim.update_frame(iEp, AX['imgScreen'],
                                          time_from_episode_start=times[i]-tEp)

        AX['cursor'].set_data(np.ones(2)*times[i], np.arange(2))
        # time
        AX['time'].set_text('     t=%.1fs\n' % times[i])
        
        return [AX['cursor'], AX['time'], AX['imgScreen'], 
                AX['imgRig'], AX['imgFace'],
                AX['imgPupil'], AX['imgWhisking'], 
                AX['pupil_fit'], 
                # AX['pupil_center'],
                AX['imgImaging'], 
                AX['imgROI1'], AX['imgROI2']]
       

    ani = animation.FuncAnimation(fig, 
                                  update,
                                  np.arange(len(times)),
                                  init_func=update,
                                  interval=100,
                                  blit=True)

    return fig, AX, ani

if __name__=='__main__':

    import argparse, physion

    parser=argparse.ArgumentParser()
    parser.add_argument("datafile", type=str)

    #######################################################
    #######################################################
    #######################################################
    parser.add_argument("--ROIs", default=[0,1,2,3,4],
                        nargs='*', type=int)
    parser.add_argument("--zoomROIs", default=[2,4],
                        nargs=2, type=int)

    parser.add_argument("--tlim", default=[10,100], 
                        nargs=2, type=float)

    #######################################################
    ###    video export   #################################
    #######################################################
    parser.add_argument("--export", action="store_true")

    parser.add_argument("--fps", 
                        type=int, default=20)
    parser.add_argument("--duration", 
                        type=float, default=0, help='video duration')
    parser.add_argument("--dpi", 
                        type=int, default=100, help='video duration')


    args = parser.parse_args()

    if args.duration>0:
        args.Ndiscret = int(args.duration*args.fps)

    if ('.nwb' in args.datafile) and os.path.isfile(args.datafile):

        exec(string_params)

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
        fig, AX, ani = draw_movie(vars(args), data)

        if args.export:
            print('writing video [...]')
            writer = animation.writers['ffmpeg'](fps=args.fps)
            ani.save('movie.mp4', writer=writer, dpi=args.dpi)

        else:
            plt.show()

    else:
        print('')
        print(' provide either a movie.py file as argument')
        print('')




