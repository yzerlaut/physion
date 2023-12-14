import xml.etree.ElementTree as ET
import numpy as np

def bruker_xml_parser(filename):
    """
    function to parse the xml metadata file produced by the Prairie software

    TODO:
    - find automated ways to count channels
    """
    mytree = ET.parse(filename)
    root = mytree.getroot()

    data = {'settings':{},
            'date':root.attrib['date'],
            'Prairie-version':root.attrib['version']}

    try:
        # to find channel names
        CHANNELS = []
        for frames in root[2:]:
            for x in frames[:4]:
                if x.tag == 'Frame':
                    for f in x:
                        if ('channelName' in f.attrib) and (f.attrib['channelName'] not in CHANNELS):
                            CHANNELS.append(f.attrib['channelName'])
    except BaseException as be:
        print(be)
        CHANNELS = ['Ch1', 'Ch2']
        print(' \n \n  /!\  Channel Names not found /!\ taking the defaults: %s \n ' % CHANNELS)

    for channel in CHANNELS:
        data[channel] = {'relativeTime':[], 'absoluteTime':[],
                         'depth_index':[], 'tifFile':[]}

    settings = root[1]
    for setting in settings:
        if 'value' in setting.attrib:
            data['settings'][setting.attrib['key']] = setting.attrib['value']
        else:
            data['settings'][setting.attrib['key']] = {}
            for s in setting:
                if s.tag == 'IndexedValue':
                    if 'description' in s.attrib:
                        data['settings'][setting.attrib['key']][s.attrib['description']] = s.attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s.attrib['value']
                elif s.tag == 'SubindexedValues':
                    if len(list(s)) == 1:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = s[0].attrib['value']
                    else:
                        data['settings'][setting.attrib['key']][s.attrib['index']] = {}
                        for sub in s:
                            data['settings'][setting.attrib['key']][s.attrib['index']][sub.attrib['description']] = [sub.attrib['value']]
    
    data['StartTime'] = root[2].attrib['time']

    if ('5.5.' in data['Prairie-version']) or ('5.6.' in data['Prairie-version']) or ('5.7.' in data['Prairie-version']):

        depths = {}
        for frames in root[2:]:
            for x in frames:
                if x.tag == 'Frame':
                    for f in x:
                        for channel in CHANNELS:
                            # print(f.attrib)
                            if f.tag == 'File' and (channel in f.attrib['channelName']):
                                data[channel]['tifFile'].append(f.attrib['filename'])
                                for key in ['relativeTime', 'absoluteTime']:
                                    data[channel][key].append(float(x.attrib[key]))
                                if len(root)>3:
                                    data[channel]['depth_index'].append(int(x.attrib['index'])-1)
                                else:
                                    data[channel]['depth_index'].append(0)
                            # depth
                            if f.tag == 'PVStateShard':
                                for d in f:
                                    if d.attrib['key']=='positionCurrent':
                                        for e in d:
                                            if e.attrib['index']=='ZAxis':
                                                for g in e:
                                                    if g.attrib['description'] not in depths:
                                                        depths[g.attrib['description']] = []
                                                    try:
                                                        depths[g.attrib['description']].append(float(g.attrib['value']))
                                                    except ValueError:
                                                        pass

        # dealing with depth  --- MANUAL for piezo plane-scanning mode because the bruker xml files don't hold this info...
        if np.sum(['Piezo' in key for key in depths.keys()]):
            Ndepth = len(np.unique(data[CHANNELS[0]]['depth_index'])) # SHOULD ALWAYS BE ODD
            try:
                for key in depths.keys():
                    if 'Piezo' in key:
                        depth_start_piezo = depths[key][0]
                depth_middle_piezo = 200 # SHOULD BE ALWAYS CENTER AT 200um
                data['depth_shift'] = np.linspace(-1, 1, Ndepth)*(depth_middle_piezo-depth_start_piezo)
            except BaseException as be:
                print(be)
                print(' /!\ plane info was not found /!\ ')
                data['depth_shift'] = np.arange(1, Ndepth+1)
        else:
            data['depth_shift'] = np.zeros(1)


    elif '5.4.' in data['Prairie-version']:

        # version without multiplabe scanning: 5.4.X
        for x in root[2]:
            for f in x:
                if f.tag == 'File':
                    channel = f.attrib['channelName']
                    data[channel]['tifFile'].append(f.attrib['filename'])
                    for key in ['relativeTime', 'absoluteTime']:
                        data[channel][key].append(float(x.attrib[key]))
                    if len(root)>3:
                        data[channel]['depth_index'].append(int(x.attrib['index'])-1)
                    else:
                        data[channel]['depth_index'].append(0)

        data['depth_shift'] = np.zeros(1)
        data['depth_index'] = np.zeros(len(data[CHANNELS[0]]['relativeTime']))

    else:

        raise NotImplementedError('\n \n  /!\  Prairie version "%s" of xml file not supported  /!\  \n ' % data['Prairie-version'])

    # ---------------------------- #
    #  translation to numpy arrays
    # ---------------------------- #
    data['Nchannels']=0
    data['channels'] = CHANNELS
    for channel in CHANNELS:
        if len(data[channel]['relativeTime'])>1:
            data['Nchannels'] += 1
        for key in ['relativeTime', 'absoluteTime']:
            data[channel][key] = np.array(data[channel][key], dtype=np.float64)
        for key in ['tifFile']:
            data[channel][key] = np.array(data[channel][key], dtype=str)

    data['Nplanes'] = len(data['depth_shift'])
                        
    return data


if __name__=='__main__':

    import sys, os, pathlib

    example_file = sys.argv[-1]
    # we test it on the example file that we have in the repo:
    # example_file = os.path.join(str(pathlib.Path(__file__).resolve().parents[2]),
                                # 'Ca_imaging', 'Bruker_xml', 'TSeries-190620-250-00-002.xml')
    
    # example_file = os.path.join(os.path.expanduser('~'), 'UNPROCESSED',
                                # 'TSeries-10202021-1256-020', 'TSeries-10202021-1256-020.xml')
    # example_file = os.path.join(os.path.expanduser('~'), 'UNPROCESSED',
    #                             'TSeries-10222021-1427-028', 'TSeries-10222021-1427-028.xml')
    
    
    data = bruker_xml_parser(example_file)
    # print(data['depth_shift'])
    # print(data.keys())
    # import pprint
    # pprint.pprint(data['settings'])
    print(data['Prairie-version'])
    for key in data['channels']:
        print('--- ', key)
        # print(data[key].keys())
        # print(data[key]['absoluteTime'][-10:])
        # print(data[key]['tifFile'][-10:])
        # print(data[key]['depth_index'][-10:])
        print(len(data[key]['relativeTime']))
    # print(data['Prairie-version'])

