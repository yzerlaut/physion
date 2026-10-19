import os
import numpy as np
import matplotlib.pylab as plt

from physion.utils import plot_tools as pt
from physion.analysis.read_NWB import Data
from physion.dataviz.raw import plot as plot_raw
from physion.dataviz.imaging import show_CaImaging_FOV


def single_recording_summary(filename, notebook_folder,
                             redo_figs=True,
                             neuropil_correction_factor=0.5,
                             Nmax_ROIs=40,
                             verbose=True):
    """
    writes the notebook page "filename.md" of a single recording
        in "notebook_folder" (and its figures in "notebook_folder/figs"):
        recording, mouse & preparation, protocol, field of view,
        full view and three 1min zooms of the raw data

    only the modalities present in the recording are shown
        (running, facemotion, pupil, opto, photodiode, visual stim, CaImaging)
    CaImaging traces: Zscore of the corrected fluorescence
        correctedFluo = rawFluo - neuropil_correction_factor x neuropil
        (all ROIs, no selection, "Nmax_ROIs" random ROIs plotted)

    with "redo_figs=False", only the markdown text is rewritten

    returns the markdown text
    """

    if verbose:
        print('running %s' % filename)
    fn = os.path.basename(filename).replace('.nwb', '')

    md = os.path.join(notebook_folder, fn+'.md')
    os.makedirs(os.path.join(notebook_folder, 'figs'), exist_ok=True)

    text = '## Recording \n\n'
    text += '-  %s \n ' % fn
    data = Data(filename)
    visual_stim_built = False
    if data.has_visual_stim():
        try:
            data.build_visual_stim() # real recording (possibly stopped)
            visual_stim_built = True
            nReal = len(data.visual_stim.experiment['time_start'])
        except Exception as be:
            # e.g. protocol files not found
            #   --> only the stim intervals are shown (no stim pictures)
            print(be)
            print(' [!!] visual stim could not be built for "%s" [!!]' % fn)
            nReal = data.nwbfile.stimulus['time_start_realigned'].num_samples
        text += '- episodes: %i \n' % nReal
    text += '\n'

    text += '## mouse & preparation \n\n'
    text += '- ID: %s  \n' % data.nwbfile.subject.subject_id
    text += '- virus: %s \n' % data.nwbfile.virus
    text += '- genotype/strain : %s / %s \n' %\
         (data.nwbfile.subject.genotype, data.nwbfile.subject.strain)
    text += '- age @rec: %s  \n' % data.nwbfile.subject.age
    if data.nwbfile.subject.date_of_birth is not None:
        text += '- DOB: %s  \n' %\
            data.nwbfile.subject.date_of_birth.strftime('%Y-%m-%d')
    else:
        text += '- DOB: N/A  \n'
    text += ' \n \n'

    text += '### Protocol (%s) \n\n' % data.metadata['protocol']
    for p in data.protocols:
        text += '- %s \n' % p
    text += ' \n'


    if data.has_ophys():

        if redo_figs:
            data.build_rawFluo(verbose=verbose)
            data.build_neuropil(verbose=verbose)
            data.build_correctedFluo(\
                    neuropil_correction_factor=neuropil_correction_factor)
            data.build_Zscore('correctedFluo')

        text += '## Field of View \n'

        if redo_figs:
            fig = plt.figure(figsize=(5., 1.5))
            AX = fig.subplots(1, 3)
            fig.subplots_adjust(left=.13, right=.96, bottom=.03, top=.98,
                                wspace=.05)

            #
            show_CaImaging_FOV(data, key='meanImg',
                            cmap=pt.get_linear_colormap('k', 'tab:green'),
                            NL=3, # non-linearity to normalize image
                            with_annotation=True,
                            ax=AX[0])
            show_CaImaging_FOV(data, key='max_proj',
                            cmap=pt.get_linear_colormap('k', 'tab:green'),
                            NL=3, # non-linearity to normalize image
                            with_annotation=False, ax=AX[1])
            show_CaImaging_FOV(data, key='meanImg',
                            cmap=pt.get_linear_colormap('k', 'tab:green'),
                            NL=3,
                            roiIndex=range(data.nROIs),
                            with_annotation=False, ax=AX[2])
            fig.savefig(os.path.join(notebook_folder, 'figs', fn+'-FOV.svg'))
            plt.close(fig)

        text += '![](figs/%s)    \n \n' % (fn+'-FOV.svg')

        text += '## Fluorescence levels \n'

        if redo_figs:
            fig, AX = plt.subplots(1, 2, figsize=(5.,1.3))
            AX[0].set_title('mean raw Fluo. ROI values')
            means = data.rawFluo.mean(axis=-1)
            AX[0].hist(means)
            pt.set_plot(AX[0], xlabel='full range', ylabel=' # ROIs')
            AX[1].hist(means[means<50])
            pt.set_plot(AX[1], xlabel='zoom in [0, 50]')
            pt.set_common_ylims(AX)
            fig.subplots_adjust(left=.12, right=.97, bottom=.32, top=.8,
                                wspace=.3)
            fig.savefig(os.path.join(notebook_folder, 'figs', fn+'-fluo.svg'))
            plt.close(fig)

        text += '![](figs/%s)    \n \n' % (fn+'-fluo.svg')

        # same random ROIs in all views
        roiIndices = np.sort(np.random.choice(np.arange(data.nROIs),
                                              np.min([Nmax_ROIs, data.nROIs]),
                                              replace=False))

    def get_settings(subsampling_factor=1,
                     with_visual_stim=False):

        settings = {}

        if data.has_running():
            settings['running'] = {'fig_fraction': 1,
                                    'subsampling': 5*subsampling_factor,
                                    'color': '#1f77b4'}
        if data.has_facemotion():
            settings['facemotion'] = {'fig_fraction': 1,
                                        'subsampling': 2*subsampling_factor,
                                        'color': 'purple'}
        if data.has_pupil():
            settings['pupil'] = {'fig_fraction': 2,
                                    'subsampling': 2*subsampling_factor,
                                    'color': '#d62728'}
        if data.has_opto():
            settings['opto'] = {'fig_fraction': 0.8,
                                'subsampling': 10*subsampling_factor,
                                'color': 'blue'}

        if data.has_ophys():
            settings['CaImaging'] = {'fig_fraction': 15,
                                    'subsampling': 1*subsampling_factor,
                                    'subquantity': 'Zscore_correctedFluo',
                                    'roiIndices': roiIndices,
                                    'color': '#2ca02c'}

        if with_visual_stim:
            if data.has_photodiode():
                settings['photodiode']= {'color': 'lightgrey', 'fig_fraction': 0.5,
                                         'subsampling': 10*subsampling_factor}
            if data.has_visual_stim():
                settings['visual_stim']= {'color': 'grey', 'fig_fraction': 0.5,
                                          'with_screen_inset': visual_stim_built}

        return settings

    # full recording view
    if redo_figs and len(get_settings())>0:
        fig = plt.figure(figsize=(5.4, 14.))
        ax = fig.add_axes([.07, .014, .908, .986]) # [left, bottom, width, height]
        plot_raw(data,
                 tlim=data.tlim,
                 settings=get_settings(subsampling_factor=20),
                 ax=ax)
        fig.savefig(os.path.join(notebook_folder, 'figs', fn+'-full.svg'))
        plt.close(fig)

    text += '## Full View: %.1fs (%.1fmin)     \n' % (data.tlim[1], data.tlim[1]/60.)
    text += '![](figs/%s)    \n \n' % (fn+'-full.svg')

    ## zoomed view
    for i, t0 in enumerate(\
        np.linspace(0, data.tlim[-1]-60,3)):
        if redo_figs and len(get_settings(with_visual_stim=True))>0:
            fig = plt.figure(figsize=(5.4, 11.6))
            ax = fig.add_axes([.07, .001, .908, .948]) # [left, bottom, width, height]
            plot_raw(data,
                     tlim=[t0, t0+60],
                     settings=get_settings(with_visual_stim=True),
                     ax=ax)
            fig.savefig(os.path.join(notebook_folder, 'figs', fn+'-%i.svg' % (i+1)))
            plt.close(fig)

        text += '## Zoom %i : 1min @ %.1fmin (%.1fs)    \n' % (i+1, t0/60., t0)
        text += '![](figs/%s)    \n\n' % (fn+'-%i.svg' % (i+1))

    data.close()

    with open(md, 'w') as f:
        f.write(text)
    return text
