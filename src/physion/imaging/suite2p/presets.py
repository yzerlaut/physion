"""
Some presets for our recording conditions
"""

presets = {\
        "Somas-only-for-zoom":{\
                               # registration
                               "nonrigid": False,
                               # detection
                               "allow_overlap":True,
                               "connected":True,
                               # neuropil
                               "inner_neuropil_radius": 4, # from 2 to 4
                               "min_neuropil_pixels": 350, 
                               # cellpose
                               "anatomical_only":2,
                               "flow_threshold": 0.1,
                               "cellprob_threshold": -2.0},\
        "Interneurons":{\
                       "nonrigid": True,
                       "spikedetect": False,
                       "allow_overlap":True,
                       "connected":True,
                       # cellpose
                       "anatomical_only":2,
                       "flow_threshold": 0.1,
                       "cellprob_threshold": 0.8},
}

if __name__=='__main__':
    for key in presets:
        print('  - ', key)
