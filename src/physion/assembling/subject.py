import os
import numpy as np
import pandas as pd


Days_per_month = [31, 28, 31, 30, 31, 30, 31,  # Jan to Jul
                  31, 30, 31, 30, 31] # Aug to Dec
def date_to_days(date):
    return 365*date[0]+Days_per_month[date[1]-1]+date[2]


def build_subject_props(args, metadata):

    subject_file = [f for f in os.listdir(args.datafolder) if '.xlsx' in f]

    subject_props = {}

    keys = ['subject_id', 'Date-of-Birth', 'description', 'age',
            'sex', 'strain', 'surgery',
            'Surgery-1', 'Date-Surgery-1', 'Surgery-2', 'Date-Surgery-2', 
            'full-genotype', 'genotype', 'virus', 'species']

    # all keys initialized to empy
    for k in keys:
        subject_props[k] = '' 

    if len(subject_file)==1:

        # load excel sheet infos
        subjectTable = pd.read_excel(os.path.join(args.datafolder, subject_file[0]),
                                     skiprows=[0])

        # Mapping from Anibio keys to NWB keys
        Mapping = {'subject_id':'Nickname', 
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

        for k in keys:
            if (k in Mapping) and (Mapping[k] in list(subjectTable.keys())):

                subject_props[k] = str(subjectTable[Mapping[k]].values[0])
                print(k, subject_props[k])

                # some cleanup already:
                # - dates
                if 'T00:00:00.000000000' in subject_props[k]:
                    subject_props[k] = np.array([int(i) for i in \
                        subject_props[k].replace('T00:00:00.000000000','').split('-')])[[0,1,2]]



    # --------------------------------------------------------------
    # read from the subject_props in metadata
    #         ---> deprecated soon...
    #                (it should be done by modifying the xslx file)
    # --------------------------------------------------------------
    elif 'subject_props' in metadata and (metadata['subject_props'] is not None):
        subject_props = metadata['subject_props']
        if 'Date-of-Birth' in subject_props:
            subject_props['Date-of-Birth'] = [int(i) for i in\
                    subject_props['Date-of-Birth'].split('/')[::-1]]
        else:
            subject_props['Date-of-Birth'] = [1988, 4, 24] # non-sense

    else:
        print('')
        print(' [!!] no subject information available [!!] ')
        print('subject_files :', subject_file)
        print('')
        
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
    # print('surgery', subject_props['surgery'])
    # -
    # ** age ** :
    recording_day = [int(i) for i in metadata['date'].split('_')]
    subject_props['age'] = 'P%iD' % (\
        date_to_days(recording_day)-date_to_days(subject_props['Date-of-Birth']))
    # print(subject_props['age'])
    # -
    # ** virus ** :
    if 'virus_dilution' in subject_props:
        subject_props['virus'] += ' (%s)' % subject_props['virus_dilution']
    # print(subject_props['virus'])


    # --------------------------------------------------------------
    # override a few properties (when curating/rebuilding datafiles)
    #         ---> deprecated soon... 
    #                (it should be done by modifying the xslx file)
    # --------------------------------------------------------------
    if hasattr(args, 'subject_id') and ('subject_id' in subject_props):
        # means we're over-writing the subject_id, we keep the old one in the description
        subject_props['description'] = 'original-subject_id=%s' % subject_props['subject_id']+\
            subject_props['description'] if ('description' in subject_props) else ''
    if hasattr(args, 'subject_id') and ('subject_id' in subject_props):
        subject_props['subject_id'] = args.subject_id
    if hasattr(args, 'genotype'):
        subject_props['genotype'] = args.genotype
    if hasattr(args, 'species'):
        subject_props['species'] = args.species
    if hasattr(args, 'virus'):
        metadata['virus'] = args.virus
    if hasattr(args, 'surgery'):
        metadata['surgery'] = args.surgery

    return subject_props
