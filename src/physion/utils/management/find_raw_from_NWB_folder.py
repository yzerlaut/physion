import physion

def find_folder_infos(args):

    DATASET = physion.analysis.read_NWB.scan_folder_for_NWBfiles(args.datafolder)

    TSeries, DayFolders, TimeFolders = [], [], []
    for i, filename in enumerate(DATASET['files'][:args.Nmax]):
        
        print('- %s' % filename)
        data = physion.analysis.read_NWB.Data(filename)

        Description = str(data.nwbfile.processing['ophys'].description)
        # print('     * %s' % Description)
        TSeries.append('TSeries-'+Description.split('TSeries-')[2].split('/')[0])
        print('     * TSeries-folder: %s' % TSeries[-1])
        DayFolders.append(str(data.nwbfile.identifier)[:10])
        TimeFolders.append(str(data.nwbfile.identifier)[11:])
        print('     * Day-folder: %s' % DayFolders[-1])
        print('     * Time-folder: %s' % TimeFolders[-1])

    return TSeries, DayFolders, TimeFolders



def build_bash_script(args):

    script = """

if ! [ -d ./to-keep ]; then mkdir ./to-keep; fi

move_to_keep() {
    # TSeries folder first
    if test -d $1; then
        {  
        mv $1 ./to-keep/
        echo $1 " moved to ./to-keep/"
        }
    else
        {
        echo ""
        echo " /!\" $1 " folder not found !!"
        echo ""
        }
    fi
    # Then VisualStim+Behavior
    if test -d $2/$3; then
        {  
        mv $2/$3 ./to-keep/
        echo $2/$3 " moved to ./to-keep/"
        }
    else
        {
        echo ""
        echo " /!\" $2/$3 " folder not found !!"
        echo ""
        }
    fi
}

"""

    TSeries, DayFolders, TimeFolders = find_folder_infos(args)

    for TSerie, DayFolder, TimeFolder in zip(TSeries, DayFolders, TimeFolders):
        
        script += 'move_to_keep %s %s %s \n' % (TSerie, DayFolder, TimeFolder)

    with open(args.script_name, 'w') as f:
        f.write(script)


if __name__=='__main__':

    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument("datafolder", type=str)
    parser.add_argument('-n', "--Nmax", type=int, 
                        help='limit the number of processed files, for debugging',
                        default=1000000)

    parser.add_argument("--script_name", type=str, default='script.sh')
    
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = parser.parse_args()

    build_bash_script(args)
