import json, argparse, tempfile, sys, os, pathlib

sys.path.append(os.path.join(pathlib.Path(__file__).resolve().parents[1], 'src'))

import physion

parser=argparse.ArgumentParser()
parser.add_argument('-p', "--protocol", help="protocol", default='scattered-moving-dots')
parser.add_argument('-b', "--buffered", help="buffer stim", action="store_true")
parser.add_argument('-i', "--index", help="stim index", type=int, default=0) 
parser.add_argument("--plot", help="plot stim", action="store_true")

args = parser.parse_args()

# get default params for protocol
protocol = physion.visual_stim.build.get_default_params(args.protocol)

protocol['buffer'] = args.buffered

if args.plot:

    import matplotlib.pylab as plt

    protocol['no-window'] = True

    stim = physion.visual_stim.build.build_stim(protocol)

    fig, ax = plt.subplots(figsize=(4,3))
    
    stim.plot_stim_picture(args.index, ax)
    
    plt.show()

else:

    stim = physion.visual_stim.build.build_stim(protocol)

    parent = physion.utils.misc.dummy_parent()

    stim.run(parent)

    stim.close()
