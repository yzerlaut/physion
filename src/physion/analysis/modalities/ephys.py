"""
electrophysiology: LFP, MUA, spikes, firing rates

methods of physion.analysis.read_NWB.Data (see the Data class)

selection of channels (e.g. a brain region, see restrict_to_region):
    - self.selected_channels: rows of the electrodes table (all by default)
    - self.selected_units: indices of the units (self.nwbfile.units)
            whose main channel is in self.selected_channels
    all ephys quantities (LFP, MUA, spikes, firing, spikeWaveforms, depths)
        are restricted to these channels/units
"""
import numpy as np
from physion.analysis import tools


def main_channel_of_waveforms(waveforms):
    """
    main channel of each unit: the channel where the variance
        of its spike waveform (over time) is the highest

    waveforms: (time, channel, unit) -> returns (unit,) channel indices
    """
    # variance over time -> (channel, unit), then the max over channels
    return np.argmax(np.var(waveforms, axis=0), axis=0)


class EphysMixin:

    def read_ephys(self):

        self.NPX_folder = \
            self.nwbfile.devices['Neuropixels OneBox'].description.split('**')[-1]

    ##############################################
    #   selection of channels and units
    ##############################################

    def has_electrodes(self):
        return getattr(self.nwbfile, 'electrodes')!=None

    def select_all_channels(self):
        """ all channels and all units of the datafile """
        self.selected_channels = np.arange(len(self.nwbfile.electrodes))\
                                    if self.has_electrodes() else None
        self.selected_units = np.arange(len(self.nwbfile.units))\
                                    if self.has_spikes() else None

    def channel_regions(self):
        """ brain region of each electrode: "location" column (Allen CCF acronyms) """
        return np.array(self.nwbfile.electrodes['location'][:]).astype(str)

    def restrict_to_region(self, region,
                           verbose=False):
        """
        restrict the ephys data to the channels of a brain region,
            e.g. data.restrict_to_region('VISp')
            region: a label of the "location" column of the electrodes
                    (or a list of labels, or None for all channels)

        sets self.selected_channels (channels of the region)
         and self.selected_units (units whose main channel is in the region)

        then re-builds the ephys quantities already built
            (LFP, MUA, spikes, firing, spikeWaveforms,
                main_channel_of_units, depth_units),
            depths relative to the top electrode of the region,
        the other ones will be restricted when they are built
        """
        if region is None:
            self.select_all_channels()
        else:
            regions = [region] if isinstance(region, str) else list(region)
            locations = self.channel_regions()
            channels = np.flatnonzero(np.isin(locations, regions))
            if len(channels)==0:
                raise ValueError('%s --> no channel in region "%s", available: %s' %\
                        (self.df_name, region, ', '.join(np.unique(locations))))
            self.selected_channels = channels

            if self.has_spikes():
                if self.has_spikeWaveforms():
                    # main channel of all units of the datafile
                    main = main_channel_of_waveforms(self.read_spikeWaveforms())
                    self.selected_units = np.flatnonzero(\
                            np.isin(main, self.selected_channels))
                else:
                    self.selected_units = np.arange(len(self.nwbfile.units))
                    print(' %s --> no "spikeWaveforms" to find the main channel of units,' %\
                            self.df_name, ' units NOT restricted to "%s" ' % region)

        if verbose:
            print(' [ok] --> restricted to "%s": %i channels, %s units ' %\
                    (region, len(self.selected_channels),
                     len(self.selected_units) if self.selected_units is not None else 'no'))

        self.rebuild_ephys(verbose=verbose)

    def rebuild_ephys(self, verbose=False):
        """ re-builds the ephys quantities already built (same time steps) """
        if hasattr(self, 'LFP'):
            self.build_LFP(verbose=verbose)
        if hasattr(self, 'MUA'):
            self.build_MUA(verbose=verbose)
        if hasattr(self, 'spikes'):
            self.build_spikes(dt=self.t_spikes[1]-self.t_spikes[0], verbose=verbose)
        if hasattr(self, 'firing'):
            self.build_firing(dt=self.t_firing[1]-self.t_firing[0], verbose=verbose)
        if hasattr(self, 'spikeWaveforms'):
            self.build_spikeWaveforms(verbose=verbose)
        if self.has_spikeWaveforms():
            self.find_main_channel_of_units(verbose=verbose)
        if hasattr(self, 'depth_units'):
            self.build_depth_units(verbose=verbose)

    ##############################################
    #   depth of the channels
    ##############################################

    def depth_of_electrodes(self, rows, key):
        """
        depth (um) of the electrodes at "rows" of the electrodes table
            0 for the top electrode of the selected channels
                (self.selected_channels, all channels of the datafile by default),
            positive below: deeper = lower probe channel index

        from the position of the contacts along the probe:
            "rel_y" column of the electrodes table ("y" in older files)

        "key" only names the quantity in the message when positions are missing
        """
        columns = self.nwbfile.electrodes.colnames
        column = 'rel_y' if 'rel_y' in columns else ('y' if 'y' in columns else None)
        if column is None:
            print(' %s --> no electrode position, "depth_%s" not available ...' %\
                    (self.df_name, key))
            return None
        position = np.array(self.nwbfile.electrodes[column][:], dtype=float)
        return position[self.selected_channels].max()-position[np.asarray(rows, dtype=int)]

    def selected_series_channels(self, key):
        """
        channels of the electrical series "key" (LFP, MUA) in self.selected_channels
            returns their indices in the data and their rows of the electrodes table
        """
        # rows of the electrodes table, in the order of the channels of the data
        rows = np.array(self.nwbfile.processing[key].data_interfaces[key].electrodes.data[:])
        indices = np.flatnonzero(np.isin(rows, self.selected_channels))
        return indices, rows[indices]

    def electrode_depths(self, key):
        """
        depth (um) of the selected channels of the electrical series "key" (LFP, MUA)
            (see depth_of_electrodes)
        """
        return self.depth_of_electrodes(self.selected_series_channels(key)[1], key)

    ##############################################
    #   LFP and MUA
    ##############################################

    def build_electrical_series(self, key):
        """
        sets self.<key> (channels, timestamps) for the selected channels,
             self.t_<key>, self.depth_<key>
             self.channels_<key>: their rows of the electrodes table
        """
        series = self.nwbfile.processing[key].data_interfaces[key]
        indices, rows = self.selected_series_channels(key)
        # we transpose to have a matrix of shape (channels, timestamps)
        setattr(self, key, np.transpose(series.data[:])[indices,:])
        setattr(self, 't_%s' % key, series.timestamps[:])
        setattr(self, 'channels_%s' % key, rows)
        setattr(self, 'depth_%s' % key, self.depth_of_electrodes(rows, key))

    def has_LFP(self):
        return ('LFP' in self.nwbfile.processing)

    def build_LFP(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=False):

        if self.has_LFP():

            self.build_electrical_series('LFP')

            if verbose:
                print(' [ok] --> "LFP" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_LFP,
                                    self.LFP[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.LFP.shape[0])])
        else:
            print(' %s --> "LFP" not available ...' % self.df_name)

    def has_MUA(self):
        return ('MUA' in self.nwbfile.processing)

    def build_MUA(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=False):

        if self.has_MUA():

            self.build_electrical_series('MUA')

            if verbose:
                print(' [ok] --> "MUA" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_MUA,
                                    self.MUA[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.MUA.shape[0])])
        else:
            print(' %s --> "MUA" not available ...' % self.df_name)

    ##############################################
    #   single units
    ##############################################

    def has_spikes(self):
        return getattr(self.nwbfile, 'units')!=None

    def build_spikes(self,
            specific_time_sampling=None,
            dt=1e-3,
            interpolation='linear',
            verbose=False):
        """
        single-unit Spikes (of the selected units)

        builds a matrix (units, times) of boolean values
            True -> means spike at that time for that unit


        by default: dt=1ms
        """
        if self.has_spikes():

            n = int((self.tlim[1]-self.tlim[0])/dt)
            self.t_spikes = np.arange(n)*dt
            self.spikes = np.zeros(\
                (len(self.selected_units), n), dtype=bool)

            for i, unit in enumerate(self.selected_units):
                for s in self.nwbfile.units['spike_times'][unit]:
                    if int(s/dt)<n:
                        self.spikes[i, int(s/dt)] = True

            if verbose:
                print(' [ok] --> "spikes" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_spikes,
                                    self.spikes[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.spikes.shape[0])])

        else:
            print(' %s --> "spikes" not available ...' % self.df_name)

    def has_firing(self):
        return getattr(self.nwbfile, 'units')!=None

    def build_firing(self,
            specific_time_sampling=None,
            dt=1e-2,
            interpolation='linear',
            verbose=False):
        """
        single-unit Spikes (of the selected units)

        builds a matrix (units, times) of firing rate values
                    ** in spikes/time-bin-duration **
            True -> means spike at that time for that unit


        by default: dt=10ms
        """
        if self.has_spikes():

            n = int((self.tlim[1]-self.tlim[0])/dt)
            self.t_firing = np.arange(n)*dt
            self.firing = np.zeros(\
                (len(self.selected_units), n), dtype=float)

            for i, unit in enumerate(self.selected_units):
                for s in self.nwbfile.units['spike_times'][unit]:
                    if int(s/dt)<n:
                        self.firing[i, int(s/dt)] += 1./dt

            if verbose:
                print(' [ok] --> "firing" built successfully ')

            if specific_time_sampling is not None:
                return np.array([\
                    tools.resample(self.t_firing,
                                    self.firing[i,:],
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)\
                                    for i in range(self.firing.shape[0])])

        else:
            print(' %s --> "spikes" not available ...' % self.df_name)

    def has_spikeWaveforms(self):
        return ('Spiking' in self.nwbfile.processing) and\
            ('single-unit Waveforms' in self.nwbfile.processing['Spiking'].data_interfaces)

    def read_spikeWaveforms(self):
        """ waveforms of all units of the datafile: (time, channel, unit) """
        return self.nwbfile.processing['Spiking'].\
                    data_interfaces['single-unit Waveforms'].features[:]

    def build_spikeWaveforms(self,
                             verbose=False):
        """
        load the spike template waveforms of the selected units
            self.spikeWaveforms: (time, channel, unit)
            on all channels: channel = row of the electrodes table
        """
        if self.has_spikeWaveforms():

            k1, k2 = 'Spiking', 'single-unit Waveforms'
            self.t_spikeWaveforms = self.nwbfile.processing[k1].data_interfaces[k2].times[:]
            self.spikeWaveforms = self.read_spikeWaveforms()[:,:,self.selected_units]

            if verbose:
                print(' [ok] --> "spikeWaveforms" built successfully ')
        else:
            print(' %s --> "spikeWaveforms" not available ...' % self.df_name)

    def find_main_channel_of_units(self,
                                   verbose=False):
        """
        main channel of each selected unit: the channel where the variance
            of its spike waveform (over time) is the highest

        sets self.main_channel_of_units: array (units,) of channel indices,
            i.e. rows of the electrodes table (self.nwbfile.electrodes),
            the channels of the waveforms
            (self.spikeWaveforms: (time, channel, unit))
        """
        if self.has_spikeWaveforms():

            if not hasattr(self, 'spikeWaveforms'):
                self.build_spikeWaveforms(verbose=verbose)

            self.main_channel_of_units = main_channel_of_waveforms(self.spikeWaveforms)

            if verbose:
                print(' [ok] --> "main_channel_of_units" found for %i units ' %\
                        len(self.main_channel_of_units))
        else:
            print(' %s --> "spikeWaveforms" not available ...' % self.df_name)

    def build_depth_units(self,
                          verbose=False):
        """
        depth (um) of each selected unit: the depth of its main channel
            (self.main_channel_of_units, see find_main_channel_of_units)
            0 for the top electrode of the selected channels, positive deeper
            -- same reference as self.depth_LFP and self.depth_MUA --

        sets self.depth_units: array (units,)
        """
        if not hasattr(self, 'main_channel_of_units'):
            self.find_main_channel_of_units(verbose=verbose)

        if hasattr(self, 'main_channel_of_units'):
            self.depth_units = self.depth_of_electrodes(self.main_channel_of_units,
                                                        'units')
            if verbose and (self.depth_units is not None):
                print(' [ok] --> "depth_units" built successfully ')
        else:
            self.depth_units = None
