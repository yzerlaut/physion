"""

[!!]
when setting up a new screen:
    --> need to also modify visual_stim/show.py
[!!]

"""
import numpy as np

SCREENS = {
    'Dell-2020':{
        'nScreens':1,
        'name':'Dell-2020',
        'screen_id':1,
        'resolution':[1280, 720],
        'width':48.0, # in cm
        'height':27.0, # in cm
        'distance_from_eye':15.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':60.,
                             'location':'top-right'},
        'gamma_correction':{'k':1.,
                            'gamma':1.},
    },
    'Dell-2020-low-resolution':{
        'nScreens':1,
        'name':'Dell-2020-low-resolution',
        'screen_id':1,
        'resolution':[320, 180],
        'width':48.0, # in cm
        'height':27.0, # in cm
        'distance_from_eye':15.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':10.,
                             'location':'top-right'},
        'gamma_correction':{'k':1.,
                            'gamma':1.},
    },
    'Lilliput':{
        'nScreens':1,
        'name':'Lilliput',
        'screen_id':1,
        'resolution':[768, 1280],
        'width':9.25, # in cm
        'height':16.25, # in cm
        'distance_from_eye':13, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':8,
                             'location':'top-right'},
        # 'gamma_correction':{'k':1.03,
                            # 'gamma':1.77},
        'gamma_correction':{'k':1.,
                            'gamma':1.},
    },
    'Mouse-Goggles':{
        'nScreens':1,
        'name':'Mouse-Googles',
        'screen_id':1,
        'resolution':[240, 210],
        'width':3.0, # in cm
        'height':3.0*240./210., # in cm
        'distance_from_eye':1.5, # in cm
        'fullscreen':True,
        'gamma_correction':{'k':1.0,
                            'gamma':1.0},
    },
    'Dell-P2018H':{
        'nScreens':1,
        'name':'Dell-P2018H',
        'screen_id':1,
        'resolution':[1280, 720],
        'width':43.4, # in cm
        'height':23.6, # in cm
        'distance_from_eye':15.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':56,
                             'location':'top-right'},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
    'LN-VR-3screens':{
        'nScreens':3,
        'name':'LN-VR-3screens',
        'screen_id':1,
        'resolution':[1024, 1280],
        'width':29.0, # in cm
        'height':34.0, # in cm
        'distance_front':17.0, # in cm
        'fullscreen':True,
        'monitoring_square':{'size':56,
                             'location':'top-right',
                             'screen_id':1},
        'gamma_correction':{'k':1.03,
                            'gamma':1.77},
    },
}

if __name__=='__main__':
    print('test')
