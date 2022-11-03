"""

"""
PREPROCESSING_SETTINGS = {
    'registration-only':{'do_registration': 1,
                         'nchannels':1,
                         'functional_chan':1,
                         'nonrigid': False,
                         'roidetect':False}, 
    'GCamp6s_1plane':{'cell_diameter':20, # in um
                     'tau':1.3,
                     'nchannels':1,
                     'functional_chan':1,
                     'align_by_chan':1,
                     'sparse_mode':False,
                     'connected':True,
                     'nonrigid':0,
                     'threshold_scaling':0.5,
                     'mask_threshold':0.3,
                     'neucoeff': 0.7}
    }

# multiplane imaging
for nplanes in [1, 3, 5, 7]:

    # for pyramidal cells
    PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes] = PREPROCESSING_SETTINGS['GCamp6s_1plane'].copy()
    PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes]['nplanes'] = nplanes

    # for interneurons
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes] = PREPROCESSING_SETTINGS['GCamp6s_%iplane' % nplanes].copy()
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['anatomical_only'] = 3
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['high_pass'] = 1
    PREPROCESSING_SETTINGS['INT_GCamp6s_%iplane' % nplanes]['nchannels'] = 2 # ASSUMES tdTomato always !!

# dealing with the specifics of the A1 settings
for key in list(PREPROCESSING_SETTINGS.keys()):
    PREPROCESSING_SETTINGS[key+'_A1'] = PREPROCESSING_SETTINGS[key].copy()
    PREPROCESSING_SETTINGS[key+'_A1']['nchannels'] = 1
    PREPROCESSING_SETTINGS[key+'_A1']['functional_chan'] = 2
    PREPROCESSING_SETTINGS[key+'_A1']['align_by_chan'] = 2

for key in list(PREPROCESSING_SETTINGS.keys()):
    PREPROCESSING_SETTINGS['2Chan_'+key] = PREPROCESSING_SETTINGS[key].copy()
    PREPROCESSING_SETTINGS['2Chan_'+key]['nchannels'] = 2
    




