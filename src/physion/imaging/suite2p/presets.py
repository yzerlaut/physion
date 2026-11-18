"""
Some presets for our recording conditions
"""

presets = {\
        # version 1.1 suite2P
        "sourcery-thy1-Taddy":{\
            "registration":{
                "nimg_init": 2000, #Number of subsampled frames for finding reference image - choose more if reference image is poor.
                "maxregshift": 0.15,
                "nonrigid":False,
                "batch_size":200,
            },
            "detection":{
                "algorithm": "sourcery",
                "highpass_time":300,
                "denoise":False,
                "sourcery_settings":{\
                    "connected":True,
                    "max_iterations":20,
                    "smooth_masks":False,
                    },
                "threshold_scaling":1.0,  #Adjust the automatically determined threshold in sparsery and sourcery by this scalar multiplier - set it smaller to find more cells.
            }
        },
        "cellpose-Interneurons":{\
            "registration":{
                "nimg_init": 2000, #Number of subsampled frames for finding reference image - choose more if reference image is poor.
                "maxregshift": 0.15, #Max allowed registration shift, as a fraction of frame max(width and height).
                "nonrigid":False, #Whether to use nonrigid registration.
                "batch_size":200, #Number of frames per batch - choose fewer if using GPU and running out of memory.
            },
            "detection":{
                "algorithm": "cellpose", #Algorithm used for cell detection ['sparsery', 'sourcery', 'cellpose'].
                "denoise":False, #Whether to use PCA denoising for cell detection.
                'cellpose_settings': {\
                    'cellpose_model': 'cpsam',
                    'cellprob_threshold': 1,#0.0 #(float, default: 0.0) specifies threshold for cell detection that will be used by cellpose.
                    'flow_threshold': 1.5, #0.4, #(float, default: 1.5) specifies flow threshold that will be used for cellpose.
                    'highpass_spatial': 0,
                    'img': 'max_proj', #meanImg',
                    'params': None,
                    #'params_chan2': None
                }
            }
        },
          "Interneurons-cibele":{\
                    # in um, sets the "cell size" box of the UI (-> suite2p "diameter" in pixels = cell_diameter / micronsPerPixel)
                    # NOT the real cell size: cellpose-SAM rescales the image by 30/diameter(px) and misses our small
                    # interneurons when they are enlarged. 48um at 1.58um/px (zoom 1, 512px) -> 30px = no rescaling.
                    # Tested on 12.11.26/h5-...-002: 20um (12px) -> 8 ROIs, 48um (30px) -> 23 ROIs.
                    # At another zoom / resolution: use 30 x micronsPerPixel (+1)
                    "cell_diameter":48,
                    "registration":{
                        "nimg_init": 2000, #Number of subsampled frames for finding reference image - choose more if reference image is poor.
                        "maxregshift": 0.15, #Max allowed registration shift, as a fraction of frame max(width and height).
                        "nonrigid":False, #Whether to use nonrigid registration.
                        "batch_size":200, #Number of frames per batch - choose fewer if using GPU and running out of memory.
                    },
                    "detection":{
                        "algorithm": "cellpose", #Algorithm used for cell detection ['sparsery', 'sourcery', 'cellpose'].
                        "denoise":False, #Whether to use PCA denoising for cell detection.
                        'cellpose_settings': {\
                            'cellpose_model': 'cpsam',
                            # LOWER -> more cells (range ~ -6 to 6, default 0.0)
                            # (-1.0 with flow_threshold 0.8 -> 37 ROIs, with a few doubles / dim spots)
                            'cellprob_threshold': 0.0,
                            # HIGHER -> more cells: masks whose shape does not match the flows are discarded (default 0.4)
                            'flow_threshold': 0.4,
                            # >0: subtract a smooth background of (value x cell diameter) before cellpose. Tested: worse here
                            'highpass_spatial': 0,
                            # 'max_proj': the interneurons are the bright spots of the max projection.
                            # NOT 'meanImg': it tiles the striped dim region with fake ROIs
                            # NOT 'max_proj / meanImg': it blows up the dark regions (registration border, dim edges)
                            'img': 'max_proj',
                            # options passed to cellpose model.eval():
                            # normalization in blocks of 100 px: evens out the bright patch and the dark regions
                            # tested (cellprob 0, flow 0.4): 1423-002: 23 -> 24 ROIs, 1335-001: 11 -> 18, 1335-002: 10 -> 20
                            # (percentile [1, 99.9]: no gain, 'sharpen_radius': 2 -> ~0 ROIs)
                            'params': {'normalize': {'tile_norm_blocksize': 100}},
                            #'params_chan2': None
                        }
                    }
                },
          # same, detection on the MEAN image: for recordings where the cells are bright in
          # the mean image but weak in the max projection (tested on 13.11.26/h5-...-1335-002: 32 ROIs vs 20)
          # but worse on others (1335-001: 12 vs 18) and fake ROIs in dark striped regions -> check the result
          "Interneurons-cibele-meanImg":{\
                    "cell_diameter":48, # see "Interneurons-cibele"
                    "registration":{
                        "nimg_init": 2000,
                        "maxregshift": 0.15,
                        "nonrigid":False,
                        "batch_size":200,
                    },
                    "detection":{
                        "algorithm": "cellpose",
                        "denoise":False,
                        'cellpose_settings': {\
                            'cellpose_model': 'cpsam',
                            'cellprob_threshold': 0.0,
                            'flow_threshold': 0.4,
                            'highpass_spatial': 0,
                            'img': 'meanImg',
                            # normalization in blocks of 100 px (see "Interneurons-cibele")
                            # tested: 1335-002: 22 -> 32 ROIs (good), 1335-001: 7 -> 12,
                            # 1423-002: 193 ROIs, almost all fake, tiling the dark striped region -> do NOT use there
                            'params': {'normalize': {'tile_norm_blocksize': 100}},
                        }
                    }
                },

    
        "Test Pyramidal 3":{\
            "frames_include":10,
            "registration":{
                "nonrigid":False
            }, 
            "detection":{
                "sparse_mode":False,
                "denoise":True,
                "spatial_scale":0,
                "allow_overlap":True,
                "max_overlap":0.5,
                "connected":True,
                "threshold_scaling":0.5,
                "algorithm": "sourcery",
                "sourcery_settings":{\
                    "threshold_scalling":0.5,
                    "diameter":10,
                    "connected":True,
                    "max_iterations":20,
                    "smooth_masks":False,
                    },
            }
        },

        #older versions suite2P 
        "Interneurons":{\
                        "nonrigid": True,
                        "allow_overlap":False,
                        "connected":True,
                        # cellpose
                        "cell_diameter":20,
                        "anatomical_only":2,
                        "flow_threshold": 0.1,
                        "cellprob_threshold": 0.8},
        "hsyn-pyramidal":{\
                        # functional detection
                        "sig_baseline":10.0,
                        'smooth_sigma':1.15,
                        'smooth_sigma_time':0,
                        'snr_thresh':1.2,
                        'soma_crop':True,
                        'sparse_mode':False,
                        'spatial_hp_cp':0,
                        'spatial_hp_detect':25,
                        'spatial_hp_reg':42,
                        'spatial_taper':40,
                        'spatscale_pix':12,
                        'subpixel':10,
                        'threshold_scaling':0.0,
                        "high_pass":300},\
                        # (ROI diameter from the "cell size" box of the UI)
        "hsyn-Ventricle-inj":{\
                        # registration
                        "nonrigid": False,
                        # functional detection
                        "sparse_mode":True,
                        "allow_overlap":False,
                        "denoise":0,
                        "connected":True,
                        "threshold_scaling":1.0,
                        "high_pass":300,
                        "spatial_hp_detect":25.0,
                        "spatial_scale":0, # no pre-shape
                        # neuropil
                        "inner_neuropil_radius": 2,
                        "min_neuropil_pixels": 350, 
                        "flow_threshold": 1.5},\
        "SomasOnly-zoomed":{\
                           # registration
                           "nonrigid": False,
                           # detection
                           "allow_overlap":True,
                           "connected":True,
                           # neuropil
                           "inner_neuropil_radius": 2,
                           "min_neuropil_pixels": 100, 
                           # cellpose
                            "spatial_hp_cp":0, # high-pass
                           "cell_diameter":20,
                           "anatomical_only":2,
                           "flow_threshold": 0.1,
                           "cellprob_threshold": -2.0},
        "hsyn-Cibele":{\
                    # functional detection
                    "sig_baseline":10.0,
                    'smooth_sigma':1.15,
                    'smooth_sigma_time':0,
                    'snr_thresh':1.2,
                    'soma_crop':True,
                    'sparse_mode':False,
                    'spatial_hp_cp':0,
                    'spatial_hp_detect':25,
                    'spatial_hp_reg':42,
                    'spatial_taper':40,
                    'spatscale_pix':12,
                    'subpixel':10,
                    'threshold_scaling':0.0,
                    "high_pass":300}
        }

SUITE2P_V1_SECTIONS = ['run', 'io', 'registration', 'detection',
                       'classification', 'extraction', 'dcnv_preprocess']

def is_v1(preset):
    """
    True if the preset is written for suite2p>=1.0, i.e. organised in
    sections ("registration", "detection", ...) as the suite2p>=1.0 settings
    -> run with "--settings settings.npy" (older presets: "--ops ops.npy")
    """
    return any(isinstance(preset.get(key), dict) for key in SUITE2P_V1_SECTIONS)

if __name__=='__main__':
    for key in presets:
        print('  - ', key)
