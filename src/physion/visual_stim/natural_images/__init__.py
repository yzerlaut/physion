"""
natural images: loading, morphing with a controlled similarity, mouse-relevant
similarity metrics, calibration and naturalness (see README.md)

    images        loading / normalizing / saving images
    preprocess    preprocessing of the images displayed by the "natural_image" stimuli
                  (histogram equalization, formerly visual_stim/preprocess_NI.py)
    spectrum      Fourier tools: periodic+smooth decomposition, spectrum equalization, scrambling
    morphing      Fourier phase morphing (PhaseMorpher), pair weights for a similarity s
    diffusion     Stable Diffusion latent morphing (DiffusionMorpher, optional dependencies:
                  not imported here, use `from physion.visual_stim.natural_images import diffusion`)
    mouse_vision  mouse acuity filter, V1 energy model, similarity metrics
    sparseness    filter-response sparseness (kurtosis) and naturalness index
    calibration   calibration of s on the V1-model similarity, export of stimulus pairs
    plots         figures

    run_phase_morphing, run_diffusion_morphing, run_naturalness: pipelines
        (python -m physion.visual_stim.natural_images.run_... --help)
"""
from . import images, preprocess, spectrum, morphing, mouse_vision, sparseness, calibration, plots
