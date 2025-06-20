"""
Some presets for our recording conditions
"""

presets = {\
        "":{},
        "hsyn-Ventricle-inj":{\
                           # registration
                           "nonrigid": False,
                           # functional detection
                           "sparse_mode":False,
                           "allow_overlap":True,
                           "denoise":0,
                           "connected":True,
                           "threshold_scaling":0.5,
                           "high_pass":300,
                           "spatial_hp_detect":25,
                           "cell_diameter":10,
                           # neuropil
                           "inner_neuropil_radius": 2,
                           "min_neuropil_pixels": 200, 
                           "flow_threshold": 0.1,
                           "cellprob_threshold": 0.5},\
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
                           "cellprob_threshold": -2.0},\
        "Interneurons":{\
                       "nonrigid": True,
                       "allow_overlap":False,
                       "connected":True,
                       # cellpose
                       "cell_diameter":20,
                       "anatomical_only":2,
                       "flow_threshold": 0.1,
                       "cellprob_threshold": 0.8},
}

if __name__=='__main__':
    for key in presets:
        print('  - ', key)
