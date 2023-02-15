"""
Script to add/modify the metadata in case of mistakes
"""
import os, tempfile, json, pathlib, shutil
import numpy as np

base_path = str(pathlib.Path(__file__).resolve().parents[1])

def update_metadata(args):

    fn = os.path.join(args.datafolder, 'metadata.npy')

    # load previous
    metadata = np.load(fn, allow_pickle=True).item()
    temp = str(tempfile.NamedTemporaryFile().name)+'.npy'
    print("""
    ---> moving the old metadata to the temporary file directory as: "%s" [...]
    """ % temp)
    shutil.move(fn, temp)

    # updates of config
    if args.config!='':
        try:
            with open(args.config) as f:
                config = json.load(f)
            metadata['config'] = args.config.split(os.path.sep)[-1].replace('.json', '')
            for key in config:
                metadata[key] = config[key]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Config" metadata failed /!\ ')

    # updates of protocol
    if args.protocol!='':
        try:
            with open(args.protocol) as f:
                protocol = json.load(f)
            metadata['protocol'] = args.protocol.split(os.path.sep)[-1].replace('.json', '')
            for key in protocol:
                metadata[key] = protocol[key]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Protocol" metadata failed /!\ ')
            
    # updates of subject
    if args.subject!='':
        print('previous subject: ', metadata['subject_ID'])
        try:
            with open(args.subject_file) as f:
                subjects = json.load(f)
            metadata['subject_ID'] = args.subject
            metadata['subject_props'] = subjects[args.subject]
        except BaseException as be:
            print(be)
            print(' /!\ update of "Subject" metadata failed /!\ ')
    
    # save new
    if 'notes' not in metadata:
        metadata['notes'] = ''

    if (args.key!='') and (args.value!=''):
        if args.key in metadata:
            print('changing "%s" from  "%s" to  "%s" in metadata ' % (args.key, metadata[args.key], args.value))
        else:
            print('creating key "%s" with value  "%s" in metadata ' % (args.key, args.value))
        metadata[args.key] = args.value
            
    np.save(fn, metadata)


if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="""
    Update metadata before building the NWB file
    """,formatter_class=argparse.RawTextHelpFormatter)
    #
    parser.add_argument('-df', "--datafolder", type=str, default='')
    #
    parser.add_argument("--from_json", action='store_true')

    # from json
    parser.add_argument('-c', "--config", type=str, default='', help='full path to a config file')
    parser.add_argument('-k', "--key", type=str, default='metadata key to change')
    parser.add_argument('-v', "--value", type=str, default='metadata value to change')
    parser.add_argument('-p', "--protocol", type=str, default='', help='full path to a protocol file')
    parser.add_argument('-sf', "--subject_file", type=str,
                        default=os.path.join(base_path, 'exp', 'subjects', 'mice_yann.json'))
    parser.add_argument('-s', "--subject", type=str, default='', help='provide the subject name')
    args = parser.parse_args()

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            update_metadata(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
