"""
Remote configuration of the Open Ephys GUI, through its HTTP server
    (Open Ephys must be running, with the server on its default port)

    python -m physion.ephys.openephys SPIKE-DETECTOR --threshold_type std --threshold_value 4

N.B. the Spike Detector plugin has no HTTP command to add/remove electrodes:
    they are only read from the settings of the signal chain. So we get the
    current settings, rewrite the electrodes, and reload the whole signal chain
    (this also re-initializes the other plugins, and requires to stop acquisition)
"""
import argparse, os, sys, tempfile, time
import xml.etree.ElementTree as ET

import requests

URL = 'http://localhost:37497/api'
LOAD_TIMEOUT = 300 # seconds, reloading the chain re-initializes the probes
OUTPUT_FOLDER = os.path.join(tempfile.gettempdir(), 'physion-openephys')

# value of "thrshlder_type" (sic) in the settings, and allowed thresholds
#       (from the parameters defined in SpikeDetector.cpp)
THRESHOLD_TYPES = {'abs':0, 'std':1, 'dyn':2}
THRESHOLD_RANGES = {'abs':(-500., -20.), # uV
                    'std':(1., 10.),     # multiple of the standard deviation
                    'dyn':(1., 10.)}     # multiple of the median


class OpenEphysError(Exception):
    pass


def get(endpoint):
    response = requests.get(URL+endpoint, timeout=5.)
    response.raise_for_status()
    return response.json()


def load_signal_chain(path):
    response = requests.put(URL+'/load', json={'path':path},
                            timeout=LOAD_TIMEOUT)
    response.raise_for_status()


def find_spike_detector(processors):
    detectors = [p for p in processors if p['name']=='Spike Detector']
    if len(detectors)!=1:
        raise OpenEphysError('expected one Spike Detector in the signal chain, found %i'\
                                    % len(detectors))
    return detectors[0]


def find_stream(processor, name):
    for stream in processor['streams']:
        if stream['name']==name:
            return stream
    raise OpenEphysError('no stream "%s" in the Spike Detector (available: %s)' % (\
            name, ', '.join(s['name'] for s in processor['streams'])))


def check_threshold(threshold_type, threshold_value):
    low, high = THRESHOLD_RANGES[threshold_type]
    if not (low<=threshold_value<=high):
        raise OpenEphysError('"%s" threshold should be in the range [%g, %g], got: %g'\
                                % (threshold_type, low, high, threshold_value))


def spike_channel_attributes(index, node_id, stream, threshold_type, threshold_value):
    """ settings of the single electrode on the contact "index" (starting at 0) """
    thresholds = {'abs':-50., 'std':4., 'dyn':4.} # defaults of the plugin
    thresholds[threshold_type] = threshold_value
    return {'name':'Electrode %i' % (index+1),
            'description':'Electrode from Spike Detector %i' % node_id,
            'num_channels':'1',
            'sample_rate':str(float(stream['sample_rate'])),
            'stream_name':stream['name'],
            'stream_source':str(stream['source_id']),
            'local_channels':str(index+1), # "local_channels" starts at 1
            'thrshlder_type':str(THRESHOLD_TYPES[threshold_type]),
            'abs_threshold1':str(thresholds['abs']),
            'std_threshold1':str(thresholds['std']),
            'dyn_threshold1':str(thresholds['dyn']),
            'waveform_type':'0'}


def find_processor_settings(settings, node_id):
    for processor in settings.iter('PROCESSOR'):
        if processor.get('nodeId')==str(node_id):
            return processor
    raise OpenEphysError('processor %i not found in the settings' % node_id)


def set_single_electrodes(settings, node_id, stream,
                          threshold_type, threshold_value):
    """
    replaces all the electrodes of the Spike Detector "node_id" (on all streams)
        by one single electrode per channel of "stream"
    """
    custom = find_processor_settings(settings, node_id).find('CUSTOM_PARAMETERS')
    for electrode in custom.findall('SPIKE_CHANNEL'):
        custom.remove(electrode)
    for i in range(stream['channel_count']):
        ET.SubElement(custom, 'SPIKE_CHANNEL',
                      spike_channel_attributes(i, node_id, stream,
                                               threshold_type, threshold_value))


def electrodes_match(settings, node_id, stream,
                     threshold_type, threshold_value):
    """ whether the electrodes of "settings" are the ones that we asked for """
    electrodes = find_processor_settings(settings, node_id).findall(\
                                            'CUSTOM_PARAMETERS/SPIKE_CHANNEL')
    if len(electrodes)!=stream['channel_count']:
        return False
    return all(\
        (e.get('local_channels')==str(i+1)) and\
        (e.get('stream_name')==stream['name']) and\
        (e.get('thrshlder_type')==str(THRESHOLD_TYPES[threshold_type])) and\
        (float(e.get('%s_threshold1' % threshold_type))==threshold_value)\
                                                for i, e in enumerate(electrodes))


def setup_spike_detector(threshold_type='std', threshold_value=4.,
                         stream_name='ProbeA', dry_run=False):

    check_threshold(threshold_type, threshold_value)
    if threshold_type=='dyn':
        print(' [!!] the "dyn" threshold may not be applied by the Spike Detector:\n'+\
              '      its source code never activates this thresholder when loading settings')

    mode = get('/status')['mode']
    if (mode!='IDLE') and not dry_run:
        raise OpenEphysError('stop the acquisition in Open Ephys first (current mode: %s)'\
                                    % mode)

    detector = find_spike_detector(get('/processors')['processors'])
    stream = find_stream(detector, stream_name)

    config = get('/config')['info']
    settings = ET.fromstring(config)
    set_single_electrodes(settings, detector['id'], stream,
                          threshold_type, threshold_value)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    stamp = time.strftime('%Y_%m_%d-%H-%M-%S')
    backup = os.path.join(OUTPUT_FOLDER, 'settings-before-%s.xml' % stamp)
    with open(backup, 'w', encoding='utf-8', newline='') as f:
        f.write(config)
    new = os.path.join(OUTPUT_FOLDER, 'settings-spike-detector-%s.xml' % stamp)
    ET.ElementTree(settings).write(new, encoding='UTF-8', xml_declaration=True)

    print(' previous settings saved as:', backup)
    print(' Spike Detector %i: %i single electrodes on "%s", threshold: %s=%g' % (\
            detector['id'], stream['channel_count'], stream['name'],
            threshold_type, threshold_value))

    if dry_run:
        print(' [dry run] settings not loaded, see:', new)
        return

    print(' reloading the signal chain [...]')
    load_signal_chain(new)

    if not electrodes_match(ET.fromstring(get('/config')['info']),
                            detector['id'], stream,
                            threshold_type, threshold_value):
        raise OpenEphysError('Open Ephys did not apply the settings, check the GUI\n'+\
                             '   (to restore the previous ones, load: %s)' % backup)
    print(' [ok] settings applied')


def main(argv=None):

    parser = argparse.ArgumentParser(prog='python -m physion.ephys.openephys',
                                     description='configure the Open Ephys GUI')
    commands = parser.add_subparsers(dest='command', required=True)

    spike = commands.add_parser('SPIKE-DETECTOR',
        help='reset the Spike Detector to one single electrode per probe contact')
    spike.add_argument('--threshold_type', choices=sorted(THRESHOLD_TYPES), default='std')
    spike.add_argument('--threshold_value', type=float, default=4.,
                       help='uV for "abs" (negative), else multiple of the noise level')
    spike.add_argument('--stream', default='ProbeA', help='name of the probe stream')
    spike.add_argument('--dry_run', action='store_true',
                       help='write the new settings to a file, without loading them')

    args = parser.parse_args(argv)

    try:
        setup_spike_detector(threshold_type=args.threshold_type,
                             threshold_value=args.threshold_value,
                             stream_name=args.stream,
                             dry_run=args.dry_run)
    except OpenEphysError as e:
        print(' [!!]', e)
        return 1
    except requests.RequestException as e:
        print(' [!!] no valid answer from the Open Ephys HTTP server\n'+\
              '      (is the GUI running?):', e)
        return 1
    return 0


if __name__=='__main__':
    sys.exit(main())
