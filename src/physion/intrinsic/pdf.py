import os, sys, pathlib
import numpy as np
from PIL import Image
import matplotlib.pylab as plt

import physion

def metadata_fig(datafolder, angle_from_axis=None):

    metadata = dict(np.load(os.path.join(datafolder, 'metadata.npy'),
                    allow_pickle=True).item())

    metadata['recording-time'] = datafolder.split(os.path.sep)[-2:]

    if angle_from_axis is not None:
        metadata['angle'] = angle_from_axis
    elif 'subject_props' in metadata:
        metadata['angle'] = metadata['subject_props']['headplate_angle_from_rig_axis_for_recording'] 
    else:
        metadata['angle'] = '' 
    
    fig, ax = plt.subplots(1, figsize=(7,1))

    string = """
    Mouse ID: "%(subject)s"

    Recorded @ %(recording-time)s

    headplate angle from rig/experimenter axis: %(angle)s
    """ % metadata

    ax.annotate(string, (0,0), size='small', xycoords='axes fraction')

    ax.axis('off')

    return fig, ax


# def show_raw_data(t, data, params, maps,
                  # pixel=(200,200)):
    
    # fig, AX = ge.figure(axes_extents=[[[5,1]],[[5,1]],[[1,1] for i in range(5)]],
                        # wspace=2.5, hspace=2.,
                        # figsize=(0.7,0.6), left=1.5, top=1.5, bottom=1)


    # AX[0][0].plot(t, data[:,pixel[0], pixel[1]], 'k', lw=1)
    # ge.set_plot(AX[0][0], ylabel='pixel\n intensity (a.u.)', xlabel='time (s)',
                # xlim=[t[0], t[-1]])
    # # ge.annotate(AX[0][0], 'pixel: %s ' % pixel, (1,1),
                # # ha='right', color='r', size='x-small')

    # AX[1][0].plot(params['STIM']['up-times'], params['STIM']['up-angle'], 'k', lw=1)
    # ge.set_plot(AX[1][0], ['left'], 
                # ylabel='bar stim.\n angle ($^o$)',
                # xlim=[t[0], t[-1]])

    # ge.image(np.rot90(maps['vasculature'], k=1), ax=AX[2][0],
             # title='green light')

    # AX[2][1].scatter([pixel[0]], [pixel[1]], s=50, color='none', edgecolor='r', lw=1)
    # ge.image(np.rot90(data[0,:,:], k=1), ax=AX[2][1],
             # title='t=%.1fs' % t[0])

    # AX[2][2].scatter([pixel[0]], [pixel[1]], s=50, color='none', edgecolor='r', lw=1)
    # ge.image(np.rot90(data[-1,:,:], k=1), ax=AX[2][2],
             # title='t=%.1fs' % t[-1])

    # spectrum = np.fft.fft(data[:,pixel[0], pixel[1]], axis=0)
    
    # power, phase = np.abs(spectrum), np.angle(spectrum)

    # AX[2][3].plot(np.arange(1, len(power)), power[1:], color=ge.gray, lw=1)
    # AX[2][3].plot([params['Nrepeat']], [power[params['Nrepeat']]], 'o', color=ge.blue, ms=4)
    # ge.annotate(AX[2][3], 'stim. freq.', (0,0.01), va='top', size='small', color=ge.blue)

    # AX[2][4].plot(np.arange(1, len(power)), phase[1:], color=ge.gray, lw=1)
    # AX[2][4].plot([params['Nrepeat']], [phase[params['Nrepeat']]], 'o', color=ge.blue, ms=4)

    # ge.set_plot(AX[2][3], ['left', 'top'], xscale='log', yscale='log', xlabelpad=4,
                # xlim=[.99,101], ylim=[power[1:].max()/120.,1.5*power[1:].max()],
                # xlabel='freq (sample unit)', ylabel='power (a.u.)')

    # ge.set_plot(AX[2][4], ['left', 'top'], xscale='log', xlabelpad=3, 
                # xlim=[.99,101], xlabel='freq.', ylabel='phase (Rd)')
    
    # return fig


def build_pdf(args, 
              angle=10,
              image_height=2.7):

    width, height = int(8.27 * 300), int(11.7 * 300) # A4 at 300dpi
    page = Image.new('RGB', (width, height), 'white')

    fig_metadata, ax = metadata_fig(args.datafolder)
    fig_metadata.savefig('/tmp/fig_metadata.png', dpi=300)
    fig = Image.open('/tmp/fig_metadata.png')
    page.paste(fig, box=(200, 160))
    fig.close()

    maps = physion.intrinsic.tools.load_maps(args.datafolder)

    # # vasculature and fluorescence image
    fig, AX = plt.subplots(1, 2, figsize=(4.5,2.6))

    if 'vasculature' in maps:
        maps['vasculature'] = (maps['vasculature']-\
                np.min(maps['vasculature']))/(np.max(maps['vasculature'])-np.min(maps['vasculature']))
        maps['vasculature'] = maps['vasculature']**args.vasc_exponent
        AX[0].imshow(maps['vasculature'], cmap='gray', vmin=0, vmax=1)
        AX[0].set_title('vasculature')


    if 'fluorescence' in maps:
        maps['fluorescence'] = (maps['fluorescence']-\
           np.min(maps['fluorescence']))/(np.max(maps['fluorescence'])-np.min(maps['fluorescence']))
        maps['fluorescence'] = maps['fluorescence']**args.fluo_exponent
        AX[1].imshow(maps['fluorescence'], cmap='gray', vmin=0, vmax=1)
        AX[1].set_title('fluorescence')

    physion.intrinsic.tools.add_arrow(AX[0], angle=args.angle_from_rig)
    physion.intrinsic.tools.add_scale_bar(AX[0], height=args.image_height)
    physion.intrinsic.tools.add_arrow(AX[1], angle=args.angle_from_rig)
    physion.intrinsic.tools.add_scale_bar(AX[1], height=args.image_height)

    AX[0].axis('off')
    AX[1].axis('off')

    fig.savefig('/tmp/fig.png', dpi=300)
    fig = Image.open('/tmp/fig.png')
    page.paste(fig, box=(int(3.4*300), int(0.1*300)))
    fig.close()

    fig_alt = physion.intrinsic.tools.plot_retinotopic_maps(maps, 'altitude')
    fig_alt.savefig('/tmp/fig_alt.png', dpi=300)

    fig_azi = physion.intrinsic.tools.plot_retinotopic_maps(maps, 'azimuth')
    fig_azi.savefig('/tmp/fig_azi.png', dpi=300)

    start, space = int(0.4*300), int(2.4*300)
    for name in ['alt', 'azi']:
        fig = Image.open('/tmp/fig_%s.png'%name)
        page.paste(fig, box=(start, space))
        # start+= fig.getbbox()[3]-fig.getbbox()[1] + 10
        start += fig.getbbox()[2]-fig.getbbox()[0]
        fig.close()

    # params, (t, data) = physion.intrinsic.tools.load_raw_data(args.datafolder, 'up')

    # fig = show_raw_data(t, data, params, maps, pixel=args.pixel)
    # fig.suptitle('example protocol: "up" ', fontsize=8)
    # fig.savefig('/tmp/fig.png', dpi=300)
    # fig = Image.open('/tmp/fig.png')
    # page.paste(fig, box=(250, int(2.8*300)))
    # fig.close()


    # trial_data = physion.intrinsic.tools.build_trial_data(maps)
    trial_data = np.load(os.path.join(args.datafolder, 'RetinotopicMappingData.npy'),
                         allow_pickle=True).item()
    trial = physion.intrinsic.RetinotopicMapping.RetinotopicMappingTrial(**trial_data)
    trial.processTrial(isPlot=False)

    for key, loc, alpha in zip(['vasculature', 'fluorescence'], [6.5, 8.9], [0.3,0.1]):

        fig, AX = plt.subplots(1, 3, figsize=(7.2,2.8))
        plt.subplots_adjust(bottom=0, wspace=0.7, right=0.95)

        fig.supylabel(key)

        if key in maps:
            AX[0].imshow(maps[key], cmap='gray', vmin=0, vmax=1)
        mean_power = maps['up-power']+maps['down-power']+maps['right-power']+maps['left-power']
        im = AX[0].imshow(mean_power, cmap=plt.cm.cool, alpha=alpha)
        AX[0].axis('off')

        fig.colorbar(im, ax=AX[0], location='top',
                     label='mean power @ stim. freq.')
        
        if key in maps:
            AX[1].imshow(maps[key], cmap='gray', vmin=0, vmax=1)
        AX[1].axis('off')

        im = AX[1].imshow(trial.signMapf, cmap=plt.cm.jet, alpha=2*alpha)
        fig.colorbar(im, ax=AX[1], location='top',
                     label='sign of retinotopic gradient')

        if key in maps:
            AX[2].imshow(maps[key], cmap='gray', vmin=0, vmax=1)
        AX[2].axis('off')

        alpha=0.1
        for key, value in trial.finalPatches.items():
            currPatch = value
            h = AX[2].imshow(currPatch.getSignedMask(),\
                    vmax=1, vmin=-1, interpolation='nearest', alpha=2*alpha, cmap='jet')
            AX[2].plot(currPatch.getCenter()[1], currPatch.getCenter()[0], '.k')

        try:
            fig.colorbar(h, ax=AX[2], location='top', ticks=[-1,1],
                         label='area segmentation')
        except BaseException as be:
            pass

        fig.savefig('/tmp/fig.png', dpi=300)
        fig = Image.open('/tmp/fig.png')
        page.paste(fig, box=(int(0.2*300), int(loc*300)))
        fig.close()

    page.save(args.output)


if __name__=='__main__':

    import argparse
    parser=argparse.ArgumentParser(description="Experiment interface",
                       formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("datafolder", type=str,default='')
    parser.add_argument("--vasc_exponent", type=float,default=1.00)
    parser.add_argument("--fluo_exponent", type=float,default=1.00)
    parser.add_argument("--angle_from_rig", type=float,default=0) # mm
    parser.add_argument("--image_height", type=float,default=2.70) # mm
    parser.add_argument("--pixel", type=int, nargs=2, default=(150,150)) 
    parser.add_argument('-o', "--output", default='fig.pdf')
    parser.add_argument('-v', "--verbose", action="store_true")
    
    args = parser.parse_args()

    build_pdf(args)
