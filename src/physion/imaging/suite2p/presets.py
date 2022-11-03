"""
From the documentatoin for the suite2p processing options:
https://suite2p.readthedocs.io/en/latest/settings.html
"""

ops0 = {
    # -------------
    # Main settings
    # -------------
    'look_one_level_down': 0.0,
    'delete_bin': False,
    'mesoscan': False,
    'bruker': True,
    'h5py': [], 'h5py_key': 'data',
    'move_bin': False, 
    'nplanes': 1, # (int, default: 1) each tiff has this many planes in sequence
    'nchannels': 1, # (int, default: 1) each tiff has this many channels per plane
    'functional_chan': 1, #  (int, default: 1) this channel is used to extract functional ROIs (1-based, so 1 means first channel, and 2 means second channel)
    'tau': 0.7, # (float, default: 1.0) The timescale of the sensor (in seconds), used for deconvolution kernel. The kernel is fixed to have this decay and is not fit to the data. We recommend: 0.7 for GCaMP6f, 1.0 for GCaMP6m, 1.25-1.5 for GCaMP6s
    'frames_include': -1, # (int, default: -1) if greater than zero, only frames_include frames are processed. useful for testing parameters on a subset of data.
    'do_bidiphase': False, # (bool, default: False) whether or not to compute bidirectional phase offset from misaligned line scanning experiment (applies to 2P recordings only). suite2p will estimate the bidirectional phase offset from ops['nimg_init'] frames if this is set to 1 (and ops['bidiphase']=0), and then apply this computed offset to all frames.
    'bidiphase': 0.0, # (int, default: 0) bidirectional phase offset from line scanning (set by user). If set to any value besides 0, then this offset is used and applied to all frames in the recording.
    'bidi_corrected': False,
    # -------------
    # Output settings
    # -------------
    'force_sktiff': False,
    'multiplane_parallel': 0.0,
    'preclassify': 0.0,
    'save_mat': False,
    'save_NWB': 1, # save as NWB output
    'combined': 1.0, # (bool, default: True) combine results across planes in separate folder 'combined' at end of processing. This folder will allow all planes to be loaded into the GUI simultaneously.
    'aspect': 1.0, # (float, default: 1.0) (**new*) ratio of um/pixels in X to um/pixels in Y (ONLY for correct aspect ratio in GUI, not used for other processing)
    # -------------
    # Frame Registration
    # -------------
    'do_registration': 1,
    'nonrigid': False, #  (bool, default: True) whether or not to perform non-rigid registration, which splits the field of view into blocks and computes registration offsets in each block separately. MOSTLY USEFUL FOR SLOW MULTIPLACE RECORDINGS !!
    'align_by_chan': 1, # (int, default: 1) which channel to use for alignment (1-based, so 1 means 1st channel and 2 means 2nd channel). If you have a non-functional channel with something like td-Tomato expression, you may want to use this channel for alignment rather than the functional channel.
    'nimg_init': 1000, # (int, default: 200) how many frames to use to compute reference image for registration
    'batch_size': 2000, # (int, default: 200) how many frames to register simultaneously in each batch. This depends on memory constraints - it will be faster to run if the batch is larger, but it will require more RAM.
    'two_step_registration': False, # (bool, default: False) whether or not to run registration twice (for low SNR data). keep_movie_raw must be True for this to work.
    'keep_movie_raw': False,
    'maxregshift': 0.1, # (float, default: 0.1) the maximum shift as a fraction of the frame size. If the frame is Ly pixels x Lx pixels, then the maximum pixel shift in pixels will be max(Ly,Lx) * ops['maxregshift'].
    'reg_tif': False, # (bool, default: False) whether or not to write the registered binary to tiff files
    'reg_tif_chan2': False, # (bool, default: False) whether or not to write the registered binary of the non-functional channel to tiff files
    'subpixel': 10,
    'smooth_sigma_time': 0.0,
    'smooth_sigma': 4.0,
    'th_badframes': 1.0,
    'pad_fft': False,
    'block_size': [256, 256], # (two ints, default: [128,128]) size of blocks for non-rigid registration, in pixels. HIGHLY recommend keeping this a power of 2 and/or 3 (e.g. 128, 256, 384, etc) for efficient fft
    'snr_thresh': 1.2,
    'maxregshiftNR': 8.0,
    '1Preg': True,
    'spatial_hp': 42,
    'spatial_hp_reg': 42.0,
    'spatial_hp_detect': 25,
    'pre_smooth': 0.0,
    'spatial_taper': 40.0,
    # -------------
    # ROI detection
    # -------------
    'roidetect': True,
    'spikedetect': True, 
    'sparse_mode': True, # (bool, default: False) whether or not to use sparse_mode cell detection
    'diameter': 12,
    'spatial_scale': 0, # (int, default: 0), what the optimal scale of the recording is in pixels. if set to 0, then the algorithm determines it automatically (recommend this on the first try). If it seems off, set it yourself to the following values: 1 (=6 pixels), 2 (=12 pixels), 3 (=24 pixels), or 4 (=48 pixels).
    'connected': True,
    'nbinned': 5000,
    'max_iterations': 20,
    'threshold_scaling': 1.0,
    'max_overlap': 0.75,
    'anatomical_only': 0,
    'high_pass': 6.0,
    'use_builtin_classifier': False,
    'inner_neuropil_radius': 2,
    'min_neuropil_pixels': 350,
    'allow_overlap': False, # back to False (25/03/22), True before because when we re-draw ROIS on top of detected ROIS -> 0 act.
    'chan2_thres': 0.5,
    'baseline': 'maximin',
    'win_baseline': 60.0,
    'sig_baseline': 10.0,
    'prctile_baseline': 8.0,
    'neucoeff': 0.7,
    'mask_threshold':0.5
}
