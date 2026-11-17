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

if __name__=='__main__':
    for key in presets:
        print('  - ', key)
