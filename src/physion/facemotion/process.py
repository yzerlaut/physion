import sys, os, pathlib, time
import numpy as np
from scipy.interpolate import interp1d
from scipy.sparse.linalg import eigsh

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from misc.progressBar import printProgressBar
from assembling.dataset import Dataset
from assembling.tools import load_FaceCamera_data
from pupil.outliers import replace_outliers
from pupil import roi


def load_folder(cls):
    """ see assembling/tools.py """
    cls.times, cls.FILES, cls.nframes, cls.Lx, cls.Ly = load_FaceCamera_data(cls.imgfolder)


def set_ROI_area(cls,
                 spatial_subsampling=1,
                 roi_coords=None):

    if (roi_coords is None) and (cls.ROI is not None):
        roi_coords = cls.ROI.position(cls)

    if roi_coords is not None:

        mx, my, sx, sy = roi_coords
        fullimg = np.load(os.path.join(cls.imgfolder, cls.FILES[0]))
        
        cls.fullx, cls.fully = np.meshgrid(np.arange(fullimg.shape[0]),
                                             np.arange(fullimg.shape[1]),
                                             indexing='ij')

        # subsampling mask NOT WORKING
        # subsampling_mask = np.zeros(cls.fullx.shape, dtype=bool)
        # subsampling_mask[::spatial_subsampling, ::spatial_subsampling]=True

        cls.zoom_cond = (cls.fullx>=mx) & (cls.fullx<=(mx+sx)) &\
            (cls.fully>=my) & (cls.fully<=my+sy) # & subsampling_mask

        # if cls.ROI is not None:
        #     mx = cls.fullx[cls.zoom_cond].min()
        #     my = cls.fully[cls.zoom_cond].min()
        #     sx = cls.fullx[cls.zoom_cond].max()-mx
        #     sy = cls.fully[cls.zoom_cond].max()-my
        #     cls.ROI.ROI.setPos([my, mx])
        #     cls.ROI.ROI.setSize([sy, sx])

        cls.Nx = cls.fullx[cls.zoom_cond].max()-cls.fullx[cls.zoom_cond].min()+1
        cls.Ny = cls.fully[cls.zoom_cond].max()-cls.fully[cls.zoom_cond].min()+1
    else:
        print('need to provide coords or to create ROI !!')


def load_ROI_data(cls, iframe1=0, iframe2=100,
                  time_subsampling=1,
                  flatten=True):

    if flatten:
        DATA = np.zeros((iframe2-iframe1, cls.Nx*cls.Ny))
    else:
        DATA = np.zeros((iframe2-iframe1, cls.Nx, cls.Ny))
    
    for frame in np.arange(iframe1, iframe2):
        fullimg = np.load(os.path.join(cls.imgfolder,
                                       cls.FILES[frame]))
        if flatten:
            DATA[frame-iframe1,:] = fullimg[cls.zoom_cond]
            # spatial subsampling in zoom cond
        else:
            DATA[frame-iframe1,:,:] = fullimg[cls.zoom_cond].reshape(cls.Nx, cls.Ny)
    
    return DATA
    



def compute_motion(cls,
                   time_subsampling=5,
                   with_ProgressBar=False):
    """

    """
    frames = np.arange(cls.nframes)[::time_subsampling]
    motion = np.zeros(len(frames)-1)

    if with_ProgressBar:
        printProgressBar(0, cls.nframes)

    for i, frame in enumerate(frames[:-1]):
        try:
            imgs = load_ROI_data(cls, frame, frame+2, flatten=True)
            motion[i] = np.mean(np.diff(imgs,axis=0)**2)
        except ValueError:
            print('problem with frame #', frame)
            pass # TO BE REMOVED !!
        
        if with_ProgressBar and (i%20==0):
            printProgressBar(frame, cls.nframes)
        
    return frames[1:], motion
    
def svdecon(X, k=100):
    """
    taken from facemap
    """
    NN, NT = X.shape
    if NN>NT:
        COV = (X.T @ X)/NT
    else:
        COV = (X @ X.T)/NN
    if k==0:
        k = np.minimum(COV.shape) - 1
    Sv, U = eigsh(COV, k = k)
    U, Sv = np.real(U), np.abs(Sv)
    U, Sv = U[:, ::-1], Sv[::-1]**.5
    if NN>NT:
        V = U
        U = X @ V
        U = U/(U**2).sum(axis=0)**.5
    else:
        V = (U.T @ X).T
        V = V/(V**2).sum(axis=0)**.5
    return U, Sv, V

def compute_SVD(cls,
                Nframe_per_chunk=1000,
                spatial_subsampling=4,
                time_subsampling=1,
                nmin_pc=250):
    """
    IN PROGRESS
    
    adapted from facemap:
    see: https://github.com/MouseLand/facemap

    compute the SVD over frames in chunks, combine the chunks and take a mega-SVD

    nmin_pc # <- how many PCs to keep in each chunk

    """
    Nframe_per_chunk = min(Nframe_per_chunk, cls.nframes)
    nsegs = int(np.floor(cls.nframes / Nframe_per_chunk)+1)

    n_PCs = min(nmin_pc, Nframe_per_chunk-1)
    
    nsegs = 1 # TROUBLESHOOTING
    for n in range(nsegs):
        i1 = n*Nframe_per_chunk
        i2 = min(cls.nframes, (n+1)*Nframe_per_chunk)
        motion = np.diff(load_ROI_data(cls, i1, i2,
                                       flatten=True), axis=0)
        avgMot = np.mean(motion, axis=0)
        U, Sv, V = svdecon(motion-avgMot)

    return U, Sv, V
        
    
if __name__=='__main__':

    import argparse, datetime

    parser=argparse.ArgumentParser()
    parser.add_argument('-df', "--datafolder", type=str,
            default='/home/yann/UNPROCESSED/2021_05_20/13-59-57/')
    parser.add_argument('-ts', "--time_subsampling", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    if args.debug:
        """
        snippet of code to design/debug the fitting algorithm
        
        ---> to be used with the "-Debug-" button of the GUI
        """

        from datavyz import ges as ge
        
        args.imgfolder = os.path.join(args.datafolder, 'FaceCamera-imgs')
        args.data = np.load(os.path.join(args.datafolder,
                                         'facemotion.npy'), allow_pickle=True).item()
        load_folder(args)
        set_ROI_area(args, roi_coords=args.data['ROI'])
        DATA = load_ROI_data(args, iframe1=0, iframe2=1000)
        frames, motion = compute_motion(args,
                                       time_subsampling=1,
                                       with_ProgressBar=True)
        ge.plot(frames, motion)
        ge.show()
        
    else:
        if os.path.isfile(os.path.join(args.datafolder, 'facemotion.npy')):
            print('Processing face motion for "%s" [...]' % os.path.join(args.datafolder, 'facemotion.npy'))
            args.imgfolder = os.path.join(args.datafolder, 'FaceCamera-imgs')
            
            args.data = np.load(os.path.join(args.datafolder,
                                             'facemotion.npy'), allow_pickle=True).item()
            load_folder(args)

            set_ROI_area(args, roi_coords=args.data['ROI'])
            frames, motion = compute_motion(args,
                                            time_subsampling=args.time_subsampling,
                                            with_ProgressBar=True)
            args.data['frame'] = frames
            args.data['t'] = args.times[frames]
            args.data['motion'] = motion
            np.save(os.path.join(args.datafolder, 'facemotion.npy'), args.data)
            print('Data successfully saved as "%s"' % os.path.join(args.datafolder, 'facemotion.npy'))
        else:
            print('  /!\ "facemotion.npy" file found, create one with the GUI  /!\ ')
