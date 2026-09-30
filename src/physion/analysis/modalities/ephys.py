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
