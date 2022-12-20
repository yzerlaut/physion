import argparse, sys, os, pathlib, datetime

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

from physion.assembling.build_NWB import build_NWB, ALL_MODALITIES

if __name__=='__main__':

    import argparse, os
    parser=argparse.ArgumentParser(description="""
    Building NWB file from mutlimodal experimental recordings
    """,formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-c', "--compression", type=int, default=0,
                        help='compression level, from 0 (no compression) to 9 (large compression, SLOW)')
    parser.add_argument('-df', "--datafolder", type=str, default='')
    parser.add_argument('-rf', "--root_datafolder", type=str, default=os.path.join(os.path.expanduser('~'), 'DATA'))
    parser.add_argument('-m', "--modalities", nargs='*', type=str, default=ALL_MODALITIES)
    parser.add_argument('-d', "--day", type=str, default=datetime.datetime.today().strftime('%Y_%m_%d'))
    parser.add_argument('-t', "--time", type=str, default='')
    parser.add_argument('-e', "--export", type=str, default='LIGHTWEIGHT',
                        help='export option [FULL / LIGHTWEIGHT / FROM_VISUALSTIM_SETUP]')
    parser.add_argument('-r', "--recursive", action="store_true")
    parser.add_argument('-v', "--verbose", action="store_true")
    parser.add_argument('-rs', "--running_sampling", default=50., type=float)
    parser.add_argument('-ps', "--photodiode_sampling", default=1000., type=float)
    parser.add_argument('-cafs', "--CaImaging_frame_sampling", default=0., type=float)
    parser.add_argument('-fcfs', "--FaceCamera_frame_sampling", default=0.001, type=float)
    parser.add_argument('-pfs', "--Pupil_frame_sampling", default=0.01, type=float)
    parser.add_argument('-sfs', "--FaceMotion_frame_sampling", default=0.005, type=float)
    parser.add_argument("--silent", action="store_true")
    parser.add_argument('-lw', "--lightweight", action="store_true")
    parser.add_argument('-fvs', "--from_visualstim_setup", action="store_true")
    parser.add_argument('-ndo', "--nidaq_only", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--standard", action="store_true")
    args = parser.parse_args()

    if not args.silent:
        args.verbose = True

    # some pre-defined settings here
    if args.export=='LIGHTWEIGHT' or args.lightweight:
        args.export='LIGHTWEIGHT'
        # 0 values for all (means 3 frame, start-middle-end)
        args.Pupil_frame_sampling = 0
        args.FaceMotion_frame_sampling = 0
        args.FaceCamera_frame_sampling = 0
        args.CaImaging_frame_sampling = 0
    if args.export=='FULL' or args.full:
        args.export='FULL'
        # push all to very high values
        args.CaImaging_frame_sampling = 1e5
        args.Pupil_frame_sampling = 1e5
        args.FaceMotion_frame_sampling = 1e5
        args.FaceCamera_frame_sampling = 0.5 # no need to have it too high
    if args.nidaq_only:
        args.export='NIDAQ'
        args.modalities = ['VisualStim', 'Electrophy']        

    if args.time!='':
        args.datafolder = os.path.join(args.root_datafolder, args.day, args.time)

    if args.datafolder!='':
        if os.path.isdir(args.datafolder):
            if (args.datafolder[-1]==os.path.sep) or (args.datafolder[-1]=='/'):
                args.datafolder = args.datafolder[:-1]
            build_NWB(args)
        else:
            print('"%s" not a valid datafolder' % args.datafolder)
    elif args.root_datafolder!='':
        FOLDERS = [l for l in os.listdir(args.root_datafolder) if len(l)==8]
        for f in FOLDERS:
            args.datafolder = os.path.join(args.root_datafolder, f)
            try:
                build_NWB(args)
            except BaseException as e:
                print(e)
