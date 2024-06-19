import datetime, os, pynwb, pathlib
from dateutil.tz import tzlocal

import numpy as np
from pynwb import NWBHDF5IO, NWBFile


def prepare_dataset(args):

    Dataset = {'files':[], 'subjects':[],'metadata':[],
               'new_subject':[],
               'old_filename':[], 
               'new_filename':[]}

    for fn in os.listdir(args.datafolder):

        if 'nwb' in fn:

            io = pynwb.NWBHDF5IO(os.path.join(args.datafolder, fn), 'r')
            nwbfile = io.read()
            
            # nasty way to get back to a python dictionary:
            metadata = eval(str(nwbfile.session_description))
            subject_name = metadata['subject_props']['Subject-ID']
            # print('- file: %s -> subject: "%s" ' % (fn, subject_name))
            Dataset['subjects'].append(subject_name)
            Dataset['files'].append(fn)

    for s, subject in enumerate(np.unique(Dataset['subjects'])):
        print('sub-%.2i/                                       (from: %s)' % (s+1, subject))
        subCond = np.array(Dataset['subjects'])==subject
        for session, fn in enumerate(np.array(Dataset['files'])[subCond]):
            new_filename = 'sub-%.2i_ses-%.2i_%s.nwb' % (s+1, session+1,
                                                         args.suffix)
            print('    %s          (from: %s)' % (new_filename, fn))
            Dataset['old_filename'].append(fn)
            Dataset['new_filename'].append(new_filename)
            Dataset['new_subject'].append('sub-%.2i' % (s+1))

    print('Dataset: N=%i mice, N=%i sessions' % (\
            len(np.unique(Dataset['subjects'])), len(Dataset['files'])))

    return Dataset

def create_new_NWB(old_NWBfile, new_NWBfile, new_subject, args):

    # read old NWB
    old_io = pynwb.NWBHDF5IO(os.path.join(args.datafolder, old_NWBfile), 'r')
    old_nwb= old_io.read()
    
    # read old NWB
    # old_io = pynwb.NWBHDF5IO(os.path.join(args.datafolder, old_NWBfile), 'r')
    # old_nwb = old_io.read()

    metadata = eval(str(old_nwb.session_description))


    new_subject = pynwb.file.Subject(description=old_nwb.subject.description,
                                    sex=old_nwb.subject.description,
                                    genotype=args.genotype if args.genotype!='' else old_nwb.genotype,
                                    species=args.species if args.species!='' else old_nwb.species,
                                    subject_id=new_subject,
                                    weight=old_nwb.subject.weight,
                                    date_of_birth=old_nwb.subject.date_of_birth)

    new_nwb = pynwb.NWBFile(identifier=old_nwb.identifier,
                            session_description=old_nwb.session_description,
                            experiment_description=old_nwb.experiment_description,
                            experimenter=old_nwb.experimenter,
                            lab=old_nwb.lab,
                            institution=old_nwb.institution,
                            notes=old_nwb.notes,
                            virus=args.virus if args.virus!='' else old_nwb.virus,
                            surgery=args.surgery if args.surgery!='' else old_nwb.surgery,
                            session_start_time=old_nwb.session_start_time,
                            subject=new_subject,
                            source_script=str(pathlib.Path(__file__).resolve()),
                            source_script_file_name=str(pathlib.Path(__file__).resolve()),
                            file_create_date=datetime.datetime.utcnow().replace(tzinfo=tzlocal()))

    for mod in old_nwb.acquisition:

        print(mod)
    for mod in old_nwb.processing:
        print(mod)

     
def build_new_dataset(Dataset, args):

    # remove folder if already existing
    if os.path.isdir(os.path.join(args.datafolder, 'curated_NWBs')):
        os.rmdir(os.path.join(args.datafolder, 'curated_NWBs'))
    # create folder for curated NWBs 
    os.mkdir(os.path.join(args.datafolder, 'curated_NWBs'))

    for iNWB, old_NWB in enumerate(Dataset['old_filename'][:args.Nmax]):

        create_new_NWB(old_NWB, 
                       Dataset['new_filename'][iNWB],
                       Dataset['new_subject'][iNWB],
                       args)


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafolder", type=str)
    parser.add_argument('-nmax', "--Nmax", type=int, 
                        help='limit the number of processed files, for debugging',
                        default=1000000)

    parser.add_argument("--suffix", type=str, default='exp')
    parser.add_argument("--genotype", type=str, default='')
    parser.add_argument("--species", type=str, default='')
    parser.add_argument("--virus", type=str, default='')
    parser.add_argument("--surgery", type=str, default='')

    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument("--transpose", 
                        help="transpose all arrays to have time as the first dimension (for data <06/2024)", 
                        action="store_true")

    args = parser.parse_args()

    Dataset = prepare_dataset(args)
    build_new_dataset(Dataset, args)


