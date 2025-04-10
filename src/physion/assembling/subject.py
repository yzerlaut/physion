import os
import numpy as np
import pandas as pd

subject_template = {\
                    'subject_id':'demo-Mouse', 
                    'Date-of-Birth':[1988, 4, 24],
                    # -
                    'description':'', 
                    # -
                    'age': 'P9999D',
                    'weight':'Unknown', 
                    'sex':'Unknown', 
                    # -
                    'strain': 'Unknown', 
                    'species': 'Mus Musculus',
                    # -
                    'full-genotype': '', 
                    'genotype': '', 
                    # -
                    'surgery': '',
                    'Surgery-1': '', 
                    'Date-Surgery-1': '', 
                    'Surgery-2': '', 
                    'Date-Surgery-2': '', 
                    # -
                    'virus': '', 
                    }

# ---------------------------------------
# Mapping from Anibio keys to NWB keys
# ---------------------------------------
Mapping = {'subject_id':'subject', 
           'full-genotype':'Lignée', 
           'genotype':'Acronyme lignée', 
           'Date-of-Birth':'D. naissance',
           'Surgery-1':'Chirurgie 1', 
           'Date-Surgery-1':'D. chirurgie 1', 
           'Surgery-2':'Chirurgie 2', 
           'Date-Surgery-2':'D. chirurgie 2', 
           'sex':'Sexe',
           'tatoo':'Marque',
           'species':'Espèce', 
           'virus':'Virus',
           'virus_dilution':'Dilution virus',
           'strain':'Souche'}


Days_per_month = [31, 28, 31, 30, 31, 30, 31,  # Jan to Jul
                  31, 30, 31, 30, 31] # Aug to Dec

def date_to_days(date):
    return 365*date[0]+np.sum(Days_per_month[:date[1]-1])+date[2]


def reformat_props(Subject, debug=False):

    subject_props = subject_template.copy()

    for k in subject_props.keys():
        if (k in Mapping) and (Mapping[k] in Subject):

            if str(Subject[Mapping[k]])!='nan':
                subject_props[k] = str(Subject[Mapping[k]])

            if debug:
                print(k, subject_props[k])

            # some cleanup already:
            # - dates
            if len(subject_props[k].split('_'))==3 and\
                    subject_props[k].split('_')[0][:2]=='20':
                subject_props[k] = [\
                        int(i) for i in subject_props[k].split('_')]

                if debug:
                    print('date -> ', k, subject_props[k])

    return subject_props



def cleanup_keys(subject_props, metadata,
                 debug=False):

    ###########################################################
    ###        Cleaning up a few keys     #####################
    ###########################################################
    # -
    # ** surgery ** :
    if subject_props['Surgery-1']!='':
        subject_props['surgery'] += '%s %s' % (\
            subject_props['Surgery-1'], subject_props['Date-Surgery-1'])
    if subject_props['Surgery-2']!='':
        subject_props['surgery'] += ' -- %s %s' % (\
            subject_props['Surgery-2'], subject_props['Date-Surgery-2'])

    if debug:
        print('surgery', subject_props['surgery'])
    # -
    # ** age ** :

    recording_day = [int(i) for i in metadata['date'].split('_')]
    age = date_to_days(recording_day)-\
                date_to_days(subject_props['Date-of-Birth'])
    subject_props['age'] = 'P%iD' % age
    if debug:
        print('age', subject_props['age'])
    # -
    # ** virus ** :
    if 'virus_dilution' in subject_props:
        subject_props['virus'] += ' (%s)' % subject_props['virus_dilution']

    if debug:
        print('virus: ', subject_props['virus'])

