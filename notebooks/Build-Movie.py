# %% [markdown]
# # Build a movie from experimental data
# assumes the following data structure
# ```
# 2020_01_01/
#         10-10-10/
#                  TSeries-010102020-001/
#                                       TSeries-010102020-001.xml
#                                       suite2p/
#                                              plane0/
#                                                    data.bin
#                  FaceCamera.mp4 # or .wmv
#                  FaceCamera-summary.npy
#                  # optional
#                  RigCamera.mp4 # or .wmv
#                  RigCamera-summary.npy
# ```
# %%
import os, sys
import numpy as np
sys.path += ['../src']
import physion

# %%
# example data from physion_Demo-Datasets:
nwbfile = os.path.expanduser(\
    '~/DATA/physion_Demo-Datasets/PYR-WT/NWBs/2025_11_14-13-54-32.nwb')
raw_data_folder =\
    os.path.expanduser('~/DATA/physion_Demo-Datasets/PYR-WT/processed/2025_11_14/13-54-32')

# %% [markdown]
# ##  Load NWB data and build data modalities

# %%
data = physion.analysis.read_NWB.Data(nwbfile)
data.init_visual_stim()
data.build_dFoF()
data.build_running_speed()

# add some smoothing for display
from scipy.ndimage import gaussian_filter1d

data.build_facemotion()
data.facemotion = gaussian_filter1d(data.facemotion, 3)
data.build_pupil_diameter()
data.pupil_diameter = gaussian_filter1d(data.pupil_diameter, 3)

# from physion.dataviz.raw import plot as plot_raw, find_default_plot_settings
# settings = find_default_plot_settings(data)
# _ = plot_raw(data, settings=settings, tlim=[0,120])

# %% [markdown]
# ##  Load FaceCamera data

# %%
NIdaq_Tstart = np.load(os.path.join(raw_data_folder, 'NIdaq.start.npy'))[0]
faceCamera = physion.utils.camera.CameraData(\
                        'FaceCamera', raw_data_folder,
                        t0=NIdaq_Tstart)


# %%
rigCamera = physion.utils.camera.CameraData(\
                        'RigCamera', raw_data_folder,
                        t0=NIdaq_Tstart)

# %% [markdown]
# ## Load Imaging data

# %%
from physion.utils.binary import BinaryFile
Ly, Lx = getattr(getattr(data.nwbfile.processing['ophys'],
                    'data_interfaces')['Backgrounds_0'], 
                    'images')['meanImg'].shape
imagingData = BinaryFile(Ly=Ly, Lx=Lx,
                        read_filename=os.path.join(\
                                raw_data_folder, data.TSeries_folder,
                                'suite2p', 'plane0','data.bin'))

# %%
params =\
{
    " ############################################## ":"",
    " ############  data sample properties ######### ":"",
    " ############################################## ":"",
    "tlim":[0,100],
    "zoomROIs":[0,1],
    "                                                ":"",
    " ############################################## ":"",
    " #############  imaging properties ############ ":"",
    " ############################################## ":"",
    "ROIs":[0,1,2,3,4,5],
    "imaging_temporal_filter":3.0,
    "imaging_spatial_filter":0.8,
    "imaging_NL":3,
    "imaging_clip":[0.3, 0.9],
    "trace_quantity":"dFoF",
    "dFoF_smoothing":0.1,
    "zoomROIs_factor":[3.0,2.5],
    "                                                ":"",
    " ############################################## ":"",
    " ##########  Face-camera properties ########### ":"",
    " ############################################## ":"",
    "Face_Lim":[0, 0, 10000, 10000],
    "Face_clip":[0.3,1.0],
    "Face_NL":3,
    "                                                ":"",
    " ############################################## ":"",
    " ##########  Rig-camera properties ############ ":"",
    " ############################################## ":"",
    "Rig_Lim":[0, 0, 10000, 10000],
    "Rig_NL":2,
    "                                                ":"",
    " ############################################## ":"",
    " ##########  annotation properties ############ ":"",
    " ############################################## ":"",
    "Tbar":5, 
    "Tbar_loc":1.0,
    "with_screen_inset":False,
    "                                                ":"",
    " ############################################## ":"",
    " ##########   layout properties  ############## ":"",
    " ############################################## ":"",
    "fractions": {"running":0.1, "running_start":0.89,
                  "whisking":0.1, "whisking_start":0.78,
                  "gaze":0.08, "gaze_start":0.7,
                  "pupil":0.15, "pupil_start":0.55,
                  "rois":0.29, "rois_start":0.29,
                  "visual_stim":2, "visual_stim_start":2.0,
                  "raster":0.28, "raster_start":0.0},
    "                                                ":""
}
# %%

time = 0

for time in [3.5, 4., 5., 6]:

    fig, AX = physion.dataviz.snapshot.layout()

    physion.dataviz.snapshot.init_imaging(AX, params, data)
    physion.dataviz.snapshot.plot_traces(AX, params, data)
    physion.dataviz.snapshot.init_screen(AX, data)
    physion.dataviz.snapshot.update_screen(AX, data, time)
    physion.dataviz.snapshot.init_camera(AX, params, faceCamera, 'Face')
    physion.dataviz.snapshot.init_camera(AX, params, rigCamera, 'Rig')
    physion.dataviz.snapshot.update_camera(AX, params, faceCamera, time, 'Face')
    physion.dataviz.snapshot.update_camera(AX, params, rigCamera, time, 'Rig')
    physion.dataviz.snapshot.init_pupil(AX, data, params, faceCamera)
    physion.dataviz.snapshot.update_pupil(AX, data, params, faceCamera, time)
    physion.dataviz.snapshot.init_whisking(AX, data, params, faceCamera)
    physion.dataviz.snapshot.update_whisking(AX, data, params, faceCamera, time)
    physion.dataviz.snapshot.update_imaging(AX, data, params, imagingData, time)
    physion.dataviz.snapshot.update_timer(AX, time)
    physion.utils.plot_tools.plt.show()

# %% [markdown]
# ## Build the movie

# %%
# fig, AX = physion.dataviz.snapshot.layout()
# _, _, ani = physion.dataviz.movie.build(fig, AX, data, params,
#                                         faceCamera=faceCamera,
#                                         rigCamera=rigCamera,
#                                         imagingData=imagingData,
#                                         Ndiscret=200)
# physion.dataviz.movie.write(ani, FPS=5)

# %%
from physion.utils import plot_tools as pt
t0 = 4.5
for t0 in np.linspace(3., 6., 12):
    i0 = np.argmin((rigCamera.times-t0)**2)
    print(i0, rigCamera.times[i0])
    fig, ax = pt.figure(ax_scale=(3,3))
    pt.matrix(
        np.rot90(rigCamera.get(i0).T, k=3),
        vmin=0,
        vmax=255,
        colormap='gray',
        ax=ax,
    )
    ax.axis('off')
    ax.set_title('t=%.1f' % t0)
# %%
import cv2 as cv

for t0 in np.linspace(3., 6., 12):
    rigCamera.cap.set(cv.CAP_PROP_POS_MSEC, t0*1000)
    fig, ax = pt.figure(ax_scale=(3,3))
    hasFrame, img = rigCamera.cap.read()
    if hasFrame:
        pt.matrix(
            # rigCamera.cap.read()[1][:,:,0].T,
            np.rot90(img[:,:,0].T, k=2),
            vmin=0,
            vmax=255,
            colormap='gray',
            ax=ax,
        )
        ax.axis('off')
        ax.set_title('t=%.1f' % t0)
# %%
