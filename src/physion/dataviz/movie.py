# all the modules from "snapshot.py"
from physion.dataviz.snapshot import *
# + all the layout is from "snapshot.py"

import matplotlib.animation as animation

def build(fig, AX, data, params,
            faceCamera=None,
            rigCamera=None,
            imagingData=None,
            Ndiscret=100):

    times = np.linspace(params['tlim'][0], params['tlim'][1], 
                        Ndiscret)
    
    # initialization
    init_imaging(AX, params, data)
    plot_traces(AX, params, data)
    init_screen(AX, data)
    init_camera(AX, params, faceCamera, 'Face')
    init_pupil(AX, data, params, faceCamera)
    init_whisking(AX, data, params, faceCamera)
    if rigCamera is not None:
        init_camera(AX, params, rigCamera, 'Rig')
    else:
        AX['imgRig'] = AX['axRig'].imshow(np.zeros((2,2)))

    def update(i=0):

        update_screen(AX, data, times[i])

        if faceCamera is not None:
            update_camera(AX, params, faceCamera, times[i], 'Face')
            update_pupil(AX, data, params, faceCamera, times[i])
            update_whisking(AX, data, params, faceCamera, times[i])

        if rigCamera is not None:
            update_camera(AX, params, rigCamera, times[i], 'Rig')

        if imagingData is not None:
            update_imaging(AX, data, params, imagingData, times[i])

        update_timer(AX, times[i])

        
        return [AX['cursor'], AX['time'], 
                AX['imgScreen'], 
                AX['imgRig'], AX['imgFace'],
                AX['imgPupil'], AX['imgWhisking'], 
                AX['pupil_fit'], 
                AX['pupil_center'],
                AX['imgImaging'], 
                AX['imgROI1'], AX['imgROI2']]
       
    ani = animation.FuncAnimation(fig, 
                                  update,
                                  np.arange(len(times)),
                                  init_func=update,
                                  interval=100,
                                  blit=True)

    return fig, AX, ani

def write(ani, 
          filename='movie.mp4',
          FPS=30.,
          DPI=100.):

    writer = animation.writers['ffmpeg'](fps=FPS)
    ani.save(filename, writer=writer, dpi=DPI)
    

# if __name__=='__main__':

#     import argparse, physion

#     parser=argparse.ArgumentParser()
#     parser.add_argument("datafile", type=str)

#     #######################################################
#     #######################################################
#     #######################################################
#     parser.add_argument("--ROIs", default=[0,1,2,3,4],
#                         nargs='*', type=int)
#     parser.add_argument("--zoomROIs", default=[2,4],
#                         nargs=2, type=int)

#     parser.add_argument("--tlim", default=[10,100], 
#                         nargs=2, type=float)

#     #######################################################
#     ###    video export   #################################
#     #######################################################
#     parser.add_argument("--export", action="store_true")

#     parser.add_argument("--fps", 
#                         type=int, default=20)
#     parser.add_argument("--duration", 
#                         type=float, default=0, help='video duration')
#     parser.add_argument("--dpi", 
#                         type=int, default=100, help='video duration')


#     args = parser.parse_args()

#     if args.duration>0:
#         args.Ndiscret = int(args.duration*args.fps)

#     if ('.nwb' in args.datafile) and os.path.isfile(args.datafile):

#         exec(string_params)

#         data = physion.analysis.read_NWB.Data(args.datafile,
#                                               with_visual_stim=True)
#         print('tlim: %s' % data.tlim)

#         root_path = os.path.dirname(args.datafile)
#         subfolder = os.path.basename(\
#                 args.datafile).replace('.nwb','')[-8:]

#          # "raw_Behavior_folder"
#         if os.path.isdir(os.path.join(root_path, subfolder)):
#             args.raw_Behavior_folder = os.path.join(root_path,
#                                                     subfolder)
#         else:
#             print(os.path.join(root_path, subfolder), 'not found')

#          # "raw_Imaging_folder"
#         if os.path.isdir(os.path.join(root_path,data.TSeries_folder)):
#             args.raw_Imaging_folder = os.path.join(root_path,
#                                                 data.TSeries_folder)
#         else:
#             print(os.path.join(root_path, data.TSeries_folder),
#                   'not found')

#         for key in params:
#             if not hasattr(args, key):
#                 setattr(args, key, params[key])
#         fig, AX, ani = draw_movie(vars(args), data)

#         if args.export:
#             print('writing video [...]')
#             writer = animation.writers['ffmpeg'](fps=args.fps)
#             ani.save('movie.mp4', writer=writer, dpi=args.dpi)

#         else:
#             plt.show()

#     else:
#         print('')
#         print(' provide either a movie.py file as argument')
#         print('')




