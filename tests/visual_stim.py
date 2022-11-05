import json, argparse, tempfile, sys 

sys.path.append('./src')

import physion

parser=argparse.ArgumentParser()
parser.add_argument("protocol", help="protocol", default='scattered-moving-dots')
parser.add_argument("-b", "--buffered", help="buffer stim", action="store_true")
parser.add_argument("-i", "--index", help="stim index", type=int, default=0) 
parser.add_argument("-p", "--plot", help="plot stim", action="store_true")

args = parser.parse_args()

protocol = physion.visual_stim.build.get_default_params(args.protocol)
physion.visual_stim.build.add_params_for_demo(protocol)

protocol['buffer'] = args.buffered

if args.plot:

    import matplotlib.pylab as plt

    params['no-window'] = True
    stim = physion.visual_stim.build.build_stim(protocol)

    fig, ax = plt.subplots()
    
    stim.plot_stim_picture(args.index, ax)
    
    plt.show()
else:

    stim = physion.visual_stim.build.build_stim(protocol)

    parent = physion.utils.misc.dummy_parent()
    stim.run(parent)

    stim.close()


