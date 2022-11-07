import imageio, os, sys, time, pathlib
import numpy as np

def create_video_folder(data_folder):
    pathlib.Path(os.path.join(data_folder, 'FaceCamera-videos')).mkdir(parents=True, exist_ok=True)

    
def export_to_video(data_folder,
                    frames_per_video = 1000,
                    video_format='avi',
                    n_frame_max=20000, i0=0, verbose=True):
    

    create_video_folder(data_folder)
    
    tstart = time.time()
    
    imgs = []
    times = np.load(os.path.join(data_folder, 'FaceCamera-times.npy'))
    t0 = np.load(os.path.join(data_folder, 'NIdaq.start.npy'))[0]
    
    n, n0, nvideo = i0, 0, 0

    file_list = os.listdir(os.path.join(data_folder, 'FaceCamera-imgs'))
    print(np.sort(file_list))
    for i in range(i0, n_frame_max):
        if os.path.isfile(os.path.join(data_folder, 'FaceCamera-imgs', '%i.npy' % i)):
            n=i
            if n>n0 and ((n-n0)%frames_per_video)==0:
                print(os.path.join(data_folder, 'FaceCamera-videos', '%i-%i.%s' % (n0, n, video_format)), 'converted !')
                imageio.mimwrite(os.path.join(data_folder, 'FaceCamera-videos', '%i-%i.%s' % (n0, n, video_format)), np.array(imgs))

                imgs = []
            imgs.append(np.load(os.path.join(data_folder, 'FaceCamera-imgs', '%i.npy' % i)))
        
    imageio.mimwrite(os.path.join(data_folder, 'FaceCamera-videos', '%i-%i.%s' % (n0, n, video_format)), np.array(imgs))
    print(os.path.join(data_folder, 'FaceCamera-videos', '%i-%i.%s' % (n0, i, video_format)), 'converted !')
    if verbose:
        print('Saving time: %.1f s ' % (time.time()-tstart))
        
    
if len(sys.argv)==1:
    print("""
    should be used as :
       python export.py test1 # to generate a set of videos from the images of the folder name "test1"
    """)
else:
    export_to_video(sys.argv[1])

    
