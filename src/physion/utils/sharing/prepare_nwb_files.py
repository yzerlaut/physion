import datetime, os, pynwb, pathlib, shutil
from dateutil.tz import tzlocal

import numpy as np
from pynwb import NWBHDF5IO, NWBFile

import physion

from physion.acquisition.recordings.Scan1Plane_Screen342V import 2P_trigger_delay


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
        print('sub-%.2i/                                       (from: %s)' % (args.subject_ID_start+s, subject))
        subCond = np.array(Dataset['subjects'])==subject
        for session, fn in enumerate(np.array(Dataset['files'])[subCond]):
            new_filename = os.path.join('sub-%.2i' % (args.subject_ID_start+s),
                                        'sub-%.2i_ses-%.2i_%s.nwb' % (args.subject_ID_start+s,
                                                                      session+1,
                                                                      args.suffix))
            print('    %s          (from: %s)' % (new_filename, fn))
            Dataset['old_filename'].append(fn)
            Dataset['new_filename'].append(new_filename)
            Dataset['new_subject'].append('sub-%.2i' % (args.subject_ID_start+s))

    print('Dataset: N=%i mice, N=%i sessions' % (\
            len(np.unique(Dataset['subjects'])), len(Dataset['files'])))

    return Dataset

def create_new_NWB(old_NWBfile, new_NWBfile, new_subject, args):

    data = physion.analysis.read_NWB.Data(os.path.join(args.datafolder, old_NWBfile),
                                          with_visual_stim=True)

    # read old NWB
    old_io = pynwb.NWBHDF5IO(os.path.join(args.datafolder, old_NWBfile), 'r')
    old_nwb= old_io.read()
    
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


    old_nwb.generate_new_id()

    for mod in old_nwb.acquisition:
        
        print('* ', mod)
        old_acq = old_nwb.acquisition[mod]

        if (type(old_acq)==pynwb.base.TimeSeries):

            new_acq = pynwb.TimeSeries(name=old_acq.name,
                                       data = np.transpose(old_acq.data) if args.transpose else old_acq.data,
                                       starting_time=old_acq.starting_time,
                                       unit=old_acq.unit,
                                       timestamps=old_acq.timestamps,
                                       rate=old_acq.rate)
            new_nwb.add_acquisition(new_acq)
            print(' [ok] ', new_acq.name)

        elif (type(old_acq)==pynwb.image.ImageSeries):

            new_acq = pynwb.image.ImageSeries(name=old_acq.name,
                                        data = old_acq.data,
                                        # data = np.transpose(old_acq.data) if args.transpose else old_acq.data,
                                       starting_time=old_acq.starting_time,
                                       unit=old_acq.unit,
                                       timestamps=old_acq.timestamps,
                                       rate=old_acq.rate)
            new_nwb.add_acquisition(new_acq)
            print(' [ok] ', new_acq.name)

        elif (type(old_acq)==pynwb.ophys.TwoPhotonSeries):

            device = new_nwb.create_device(
                name='Microscope', 
                description='2P@ICM',
                manufacturer='Bruker')
            optical_channel = pynwb.ophys.OpticalChannel(
                                name='OpticalChannel', 
                                description='an optical channel', 
                                emission_lambda=500.)
            imaging_plane = new_nwb.create_imaging_plane(
                    name=old_acq.imaging_plane.name,
                    optical_channel=optical_channel,
                    imaging_rate=old_acq.imaging_plane.imaging_rate,
                    description=old_acq.imaging_plane.description,
                    device=device,
                    excitation_lambda=old_acq.imaging_plane.excitation_lambda,
                    indicator=old_acq.imaging_plane.indicator,
                    location=old_acq.imaging_plane.location,
                    grid_spacing=old_acq.imaging_plane.grid_spacing,
                    grid_spacing_unit='microns')

            
            max_proj = np.array(old_nwb.processing['ophys'].data_interfaces['Backgrounds_0'].images['max_proj'])
            image_series = pynwb.ophys.TwoPhotonSeries(name=old_acq.name,
                                                       dimension=[2], 
                                                       data=np.array([max_proj]),
                                                       imaging_plane=imaging_plane, 
                                                       unit='s', 
                                                       timestamps=[0],
                                                       comments=old_acq.comments)
    
            new_nwb.add_acquisition(image_series)

        else:
            print(' X ', mod, '   / !! \\ ')
            print(old_acq)


    for param in old_nwb.stimulus:
        nt = len(old_nwb.stimulus[param].timestamps[:])
        if old_nwb.stimulus[param].data[:].shape[0]==1:
            array = np.ones((nt,1))*old_nwb.stimulus[param].data[0]
        else:
            array = np.reshape(old_nwb.stimulus[param].data[:nt],
                              (nt,1))
        VisualStimProp = pynwb.TimeSeries(name=param,
                    data = array,
                    unit='seconds',
                    timestamps=old_nwb.stimulus[param].timestamps[:])
        new_nwb.add_stimulus(VisualStimProp)


    for mod in old_nwb.processing:
        
        old_proc = old_nwb.processing[mod]

        new_proc = new_nwb.create_processing_module(name=old_proc.name,
                                                    description=old_proc.description)
        print(' - ', new_proc.name)

        if mod=='ophys':

            # we create the image segmentation
            img_seg = pynwb.ophys.ImageSegmentation()
            ps = img_seg.create_plane_segmentation(
                name='PlaneSegmentation',
                description='suite2p output',
                imaging_plane=imaging_plane,
                reference_images=image_series)
            new_proc.add(img_seg)
            
            ncells = old_nwb.processing['ophys'].data_interfaces['Fluorescence']['Fluorescence'].data.shape[1]
        
            for n in np.arange(ncells):
                ps.add_roi(\
                        pixel_mask=\
                            tuple(old_nwb.processing['ophys'].data_interfaces['ImageSegmentation']['PlaneSegmentation']['pixel_mask'][n]))

            rt_region = ps.create_roi_table_region(
                region=list(np.arange(0, ncells)),
                description='all ROIs')

            ps.add_column('plane', 'one column - plane ID',
                old_nwb.processing['ophys'].data_interfaces['ImageSegmentation']['PlaneSegmentation']['plane'].data[:])

        for key in old_proc.data_interfaces:

            print(' - ', mod, ', ', key)

            if mod=='ophys' and (key in ['Fluorescence', 'Neuropil']):

                old_RRS = old_proc.data_interfaces[key]
                    
                if old_RRS[key].timestamps[0]<2P_trigger_delay:
                    print(" \n / ! \ the 2P-trigger-delay was ommited, adding it ! / ! \ \n ")
                    new_timestamps = old_RRS[key].timestamps[:]+2P_trigger_delay
                else:
                    new_timestamps = old_RRS[key].timestamps[:]

                roi_resp_series = pynwb.ophys.RoiResponseSeries(
                    name=old_RRS.name,
                    data=np.array(old_RRS[key].data),
                    rois=rt_region,
                    timestamps=new_timestamps,
                    unit='lumens')
                fl = pynwb.ophys.Fluorescence(roi_response_series=roi_resp_series,
                                              name=key)
                new_proc.add(fl)

            elif 'Background' in key:

                images = pynwb.base.Images(key)
                for image in old_proc.data_interfaces[key].images:
                    images.add_image(pynwb.image.GrayscaleImage(name=image,
                                                    data=old_proc.data_interfaces[key].images[image].data[:]))
                new_proc.add(images)

            elif mod!='ophys':
                if args.transpose and\
                        hasattr(old_proc.data_interfaces[key], 'timestamps'):
                    old_TS = old_proc.data_interfaces[key]
                    new_TS = pynwb.TimeSeries(name=old_TS.name,
                                          data = np.transpose(old_TS.data) if args.transpose else np.array(old_TS.data),
                                          starting_time=old_TS.starting_time,
                                          unit=old_TS.unit,
                                              timestamps=old_TS.timestamps[:],
                                          rate=old_TS.rate)
                    new_proc.add(new_TS)
                else:
                    new_proc.add(old_proc.data_interfaces[key])

    if args.verbose:
        print(10*'\n')
        print(old_nwb)
        print(10*'\n')
        print(new_nwb)

    filename = os.path.join(args.datafolder, 'curated_NWBs', new_NWBfile)
    io = pynwb.NWBHDF5IO(filename, mode='w')
    print("""
    ---> Creating the NWB file: "%s"
    """ % filename)
    new_nwb.generate_new_id()
    io.write(new_nwb, link_data=False)
    io.close()
     
def build_new_dataset(Dataset, args):

    # remove folder if already existing
    if os.path.isdir(os.path.join(args.datafolder, 'curated_NWBs')):
        shutil.rmtree(os.path.join(args.datafolder, 'curated_NWBs'))

    # create folder for curated NWBs 
    os.mkdir(os.path.join(args.datafolder, 'curated_NWBs'))

    for iNWB, old_NWB in enumerate(Dataset['old_filename'][:args.Nmax]):

        df = os.path.dirname(os.path.join(args.datafolder, 'curated_NWBs',
                                Dataset['new_filename'][iNWB]))

        # create folder
        pathlib.Path(df).mkdir(parents=True, exist_ok=True)

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

    parser.add_argument("--subject_ID_start", type=int, default=1)
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


