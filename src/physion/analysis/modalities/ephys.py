"""
electrophysiology: LFP, MUA, spikes, firing rates

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
from physion.analysis import tools


class EphysMixin:

    def read_ephys(self):
       
        self.NPX_folder = \
            self.nwbfile.devices['Neuropixels OneBox'].description.split('**')[-1]

    def depth_of_electrodes(self, rows, key):
        """
        depth (um) of the electrodes at "rows" of the electrodes table
            0 for the top electrode of the datafile (electrodes table),
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
        return position.max()-position[np.asarray(rows)]

    def electrode_depths(self, key):
        """
        depth (um) of the channels of the electrical series "key" (LFP, MUA)
            (see depth_of_electrodes)
        """
        # rows of the electrodes table, in the order of the channels of the data
        rows = np.array(self.nwbfile.processing[key].data_interfaces[key].electrodes.data[:])
        return self.depth_of_electrodes(rows, key)

    def has_LFP(self):
        return ('LFP' in self.nwbfile.processing)

    def build_LFP(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=False):

        if self.has_LFP():

            # we transpose to have a matrix of shape (channels, timestamps)
            self.LFP = np.transpose(\
                self.nwbfile.processing['LFP'].data_interfaces['LFP'].data[:])
            self.t_LFP = self.nwbfile.processing['LFP'].data_interfaces['LFP'].timestamps[:]
            self.depth_LFP = self.electrode_depths('LFP')

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

            # we transpose to have a matrix of shape (channels, timestamps)
            self.MUA = np.transpose(\
                self.nwbfile.processing['MUA'].data_interfaces['MUA'].data[:])
            self.t_MUA = self.nwbfile.processing['MUA'].data_interfaces['MUA'].timestamps[:]
            self.depth_MUA = self.electrode_depths('MUA')

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

    def has_spikes(self):
        return getattr(self.nwbfile, 'units')!=None

    def build_spikes(self,
            specific_time_sampling=None,
            dt=1e-3,
            interpolation='linear',
            verbose=False):
        """
        single-unit Spikes

        builds a matrix (units, times) of boolean values
            True -> means spike at that time for that unit


        by default: dt=1ms
        """
        if self.has_spikes():

            n = int((self.tlim[1]-self.tlim[0])/dt)
            self.t_spikes = np.arange(n)*dt
            self.spikes = np.zeros(\
                (len(self.nwbfile.units), n), dtype=bool)
            
            for i, unit in enumerate(self.nwbfile.units):
                for s in unit.spike_times.values[:][0]:
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
        single-unit Spikes

        builds a matrix (units, times) of firing rate values 
                    ** in spikes/time-bin-duration **
            True -> means spike at that time for that unit


        by default: dt=10ms
        """
        if self.has_spikes():

            n = int((self.tlim[1]-self.tlim[0])/dt)
            self.t_firing = np.arange(n)*dt
            self.firing = np.zeros(\
                (len(self.nwbfile.units), n), dtype=float)
            
            for i, unit in enumerate(self.nwbfile.units):
                for s in unit.spike_times.values[:][0]:
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

    def build_spikeWaveforms(self,
                             verbose=False):
        """ 
        load the spike template waveforms 
        """
        if self.has_spikeWaveforms():

            k1, k2 = 'Spiking', 'single-unit Waveforms'
            self.t_spikeWaveforms = self.nwbfile.processing[k1].data_interfaces[k2].times[:]
            self.spikeWaveforms = self.nwbfile.processing[k1].data_interfaces[k2].features[:]

            if verbose:
                print(' [ok] --> "spikeWaveforms" built successfully ')
        else:
            print(' %s --> "spikeWaveforms" not available ...' % self.df_name)

    def find_main_channel_of_units(self,
                                   verbose=False):
        """
        main channel of each single unit: the channel where the variance
            of its spike waveform (over time) is the highest

        sets self.main_channel_of_units: array (units,) of channel indices,
            i.e. rows of the electrodes table (self.nwbfile.electrodes),
            the channels of the waveforms
            (self.spikeWaveforms: (time, channel, unit))
        """
        if self.has_spikeWaveforms():

            if not hasattr(self, 'spikeWaveforms'):
                self.build_spikeWaveforms(verbose=verbose)

            # variance over time -> (channel, unit), then the max over channels
            variance = np.var(self.spikeWaveforms, axis=0)
            self.main_channel_of_units = np.argmax(variance, axis=0)

            if verbose:
                print(' [ok] --> "main_channel_of_units" found for %i units ' %\
                        len(self.main_channel_of_units))
        else:
            print(' %s --> "spikeWaveforms" not available ...' % self.df_name)

    def build_depth_units(self,
                          verbose=False):
        """
        depth (um) of each single unit: the depth of its main channel
            (self.main_channel_of_units, see find_main_channel_of_units)
            0 for the top electrode of the datafile, positive deeper
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
