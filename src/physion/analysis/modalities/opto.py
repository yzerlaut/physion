"""
optogenetic stimulation

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
from physion.analysis import tools


class OptoMixin:

    def has_opto(self):
        return ('OptogeneticSeries' in self.nwbfile.stimulus)

    def build_opto(self,
        specific_time_sampling=None,
        interpolation='linear',
        verbose=True):

        if self.has_opto():

            self.opto = self.nwbfile.stimulus['OptogeneticSeries'].data[:]

            self.t_opto = tools.build_timestamps(\
                        self.nwbfile.stimulus, 'OptogeneticSeries')

            if verbose:
                print(' [ok] --> "opto" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_opto,
                                    self.opto,
                                    specific_time_sampling,
                                    interpolation=interpolation,
                                    verbose=verbose)
        else:
            print(' %s --> "opto" not available ...' % self.df_name)
