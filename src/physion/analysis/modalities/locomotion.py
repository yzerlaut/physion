"""
locomotion (running speed)

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
from physion.analysis import tools


class LocomotionMixin:

    def has_running(self):
        return ('Running-Speed' in self.nwbfile.acquisition)

    def build_running(self,
                      specific_time_sampling=None,
                      interpolation='linear',
                      verbose=False, 
                      absolute=True):
        """
        Build running speed from NWB acquisition.
        
        Parameters
        ----------
        specific_time_sampling : array-like or None - If provided, resample running speed to these time points.
        interpolation : str - Interpolation method for resampling ('linear', 'nearest', etc.).
        verbose : bool - If True, print progress messages.
        absolute : bool - If True, running speed values are converted to absolute values.

        Returns
        -------
        running_resampled : np.ndarray - Resampled running speed if specific_time_sampling is provided, otherwise None.
        """
        if self.has_running():
            if absolute:
                self.running = np.abs(self.nwbfile.acquisition['Running-Speed'].data[:, 0])
            else : 
                self.running = self.nwbfile.acquisition['Running-Speed'].data[:, 0]

            self.t_running = tools.build_timestamps(\
                        self.nwbfile.acquisition, 'Running-Speed')

            if verbose:
                print(' [ok] --> "running" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_running,
                                    self.running,
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)

        else:
            print(' %s --> "running" not available ...' % self.df_name)
