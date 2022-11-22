import datetime, os
import numpy as np
from natsort import natsorted 

from pynwb import NWBFile
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.device import Device
from pynwb.ophys import OpticalChannel
from pynwb.ophys import TwoPhotonSeries
from pynwb.ophys import ImageSegmentation
from pynwb.ophys import RoiResponseSeries
from pynwb.ophys import Fluorescence
from pynwb import NWBHDF5IO

def add_ophys_processing_from_suite2p(save_folder, nwbfile, xml, 
                                      device=None,
                                      optical_channel=None,
                                      imaging_plane=None,
                                      image_series=None):
    """ 
    adapted from suite2p/suite2p/io/nwb.py "save_nwb" function
    """

    plane_folders = natsorted([ f.path for f in os.scandir(save_folder) if f.is_dir() and f.name[:5]=='plane'])
    OPS = [np.load(os.path.join(f, 'ops.npy'), allow_pickle=True).item() for f in plane_folders]

    if len(OPS)>1:
        multiplane, nplanes = True, len(plane_folders)
        pData_folder = os.path.join(save_folder, 'combined') # processed data folder -> using the "combined output from suite2p"
    else:
        multiplane, nplanes = False, 1
        pData_folder = os.path.join(save_folder, 'plane0') # processed data folder

    # find time sampling per plane
    functional_chan = ('Ch1' if len(xml['Ch1']['relativeTime'])>1 else 'Ch2') # functional channel is one of the two !!
    CaImaging_timestamps = xml[functional_chan]['relativeTime']+float(xml['settings']['framePeriod'])/2.

    ops = np.load(os.path.join(pData_folder, 'ops.npy'), allow_pickle=True).item() 
    
    if device is None:
        device = nwbfile.create_device(
            name='Microscope', 
            description='My two-photon microscope',
            manufacturer='The best microscope manufacturer')
    if optical_channel is None:
        optical_channel = OpticalChannel(
            name='OpticalChannel', 
            description='an optical channel', 
            emission_lambda=500.)
    if imaging_plane is None:
        imaging_plane = nwbfile.create_imaging_plane(
            name='ImagingPlane',
            optical_channel=optical_channel,
            imaging_rate=ops['fs'],
            description='standard',
            device=device,
            excitation_lambda=600.,
            indicator='GCaMP',
            location='V1',
            grid_spacing=([2,2,30] if multiplane else [2,2]),
            grid_spacing_unit='microns')

    if image_series is None:
        # link to external data
        image_series = TwoPhotonSeries(
            name='TwoPhotonSeries', 
            dimension=[ops['Ly'], ops['Lx']],
            external_file=(ops['filelist'] if 'filelist' in ops else ['']), 
            imaging_plane=imaging_plane,
            starting_frame=[0], 
            format='external', 
            starting_time=0.0, 
            rate=ops['fs'] * ops['nplanes']
        )
        nwbfile.add_acquisition(image_series) # otherwise, were added

    # processing
    img_seg = ImageSegmentation()
    ps = img_seg.create_plane_segmentation(
        name='PlaneSegmentation',
        description='suite2p output',
        imaging_plane=imaging_plane,
        reference_images=image_series
    )
    ophys_module = nwbfile.create_processing_module(
        name='ophys', 
        description='optical physiology processed data\n TSeries-folder=%s' % save_folder)
    ophys_module.add(img_seg)

    # file_strs = ['F_chan2.npy', 'Fneu_chan2.npy', 'spks.npy']
    file_strs = ['F.npy', 'Fneu.npy', 'spks.npy']
    traces = []

    iscell = np.load(os.path.join(pData_folder, 'iscell.npy')).astype(bool)

    if ops['nchannels']>1:
        if os.path.isfile(os.path.join(pData_folder, 'redcell_manual.npy')):
            redcell = np.load(os.path.join(pData_folder, 'redcell_manual.npy'))[iscell[:,0], :]
        else:
            print('\n'+30*'--')
            print(' /!\ no file found for the manual labelling of red cells (generate it with the red-cell labelling GUI) /!\ ')
            print(' /!\ taking the raw suit2p output with the classifier settings /!\ ')
            print('\n'+30*'--')
            redcell = np.load(os.path.join(pData_folder, 'redcell.npy'))[iscell[:,0], :]
            
    for fstr in file_strs:
        traces.append(np.load(os.path.join(pData_folder, fstr))[iscell[:,0], :].T) # transposing the traces to fill the NWB requirements ! (Ntime_samples, Nrois)
        
    stat = np.load(os.path.join(pData_folder, 'stat.npy'), allow_pickle=True)

    ncells = np.sum(iscell[:,0])
    plane_ID = np.zeros(ncells)
    for n in np.arange(ncells):
        pixel_mask = np.array([stat[iscell[:,0]][n]['ypix'], stat[iscell[:,0]][n]['xpix'], 
                               stat[iscell[:,0]][n]['lam']])
        ps.add_roi(pixel_mask=pixel_mask.T)
        if 'iplane' in stat[0]:
            plane_ID[n] = stat[iscell[:,0]][n]['iplane']

    if ops['nchannels']>1:
        ps.add_column('redcell', 'two columns - redcell & probcell', redcell)
    ps.add_column('plane', 'one column - plane ID', plane_ID)

    rt_region = ps.create_roi_table_region(
        region=list(np.arange(0, ncells)),
        description='all ROIs')

    # FLUORESCENCE (all are required) /!\ YANN: removed spks.npy
    file_strs = ['F.npy', 'Fneu.npy']
    name_strs = ['Fluorescence', 'Neuropil']

    for i, (fstr,nstr) in enumerate(zip(file_strs, name_strs)):
        roi_resp_series = RoiResponseSeries(
            name=nstr,
            data=traces[i],
            rois=rt_region,
            unit='lumens',
            timestamps=CaImaging_timestamps[::nplanes]) # ideally should be shifted for each ROI depending on the plane...
        fl = Fluorescence(roi_response_series=roi_resp_series, name=nstr)
        ophys_module.add(fl)

    # BACKGROUNDS
    # (meanImg, Vcorr and max_proj are REQUIRED)
    bg_strs = ['meanImg', 'meanImgE', 'Vcorr', 'max_proj', 'meanImg_chan2']
    nplanes = ops['nplanes']
    for iplane in range(nplanes):
        images = Images('Backgrounds_%d'%iplane)
        for bstr in bg_strs:
            if bstr in ops:
                if bstr=='Vcorr' or bstr=='max_proj':
                    img = np.zeros((ops['Ly'], ops['Lx']), np.float32)
                    img[ops['yrange'][0]:ops['yrange'][-1], 
                        ops['xrange'][0]:ops['xrange'][-1]] = ops[bstr]
                else:
                    img = ops[bstr]
                images.add_image(GrayscaleImage(name=bstr, data=img))

        ophys_module.add(images)


if __name__=='__main__':

    # creat dummy example

    import argparse, os, pynwb, datetime
    from physion.assembling.saving import get_files_with_extension, get_TSeries_folders
    from physion.assembling.IO.bruker_xml_parser import bruker_xml_parser
    
    parser=argparse.ArgumentParser(description="""
    Building test NWB file with Ophys data
    """,formatter_class=argparse.RawTextHelpFormatter)
    # main
    parser.add_argument("CaImaging_folder", type=str, default='')
    # other
    args = parser.parse_args()

    nwbfile = pynwb.NWBFile('Intrinsic Imaging data following bar stimulation',
                            'intrinsic',
                            datetime.datetime.utcnow(),
                            file_create_date=datetime.datetime.utcnow())

    CaFn = get_files_with_extension(args.CaImaging_folder, extension='.xml')[0]# get Tseries metadata
    xml = bruker_xml_parser(CaFn) # metadata

    add_ophys_processing_from_suite2p(os.path.join(args.CaImaging_folder, 'suite2p'),
                                      nwbfile, xml) # ADD UPDATE OF starting_time

    
    





