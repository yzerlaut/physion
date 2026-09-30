"""
face camera: pupil, gaze and facemotion

methods of physion.analysis.read_NWB.Data (see the Data class)
"""
import numpy as np
from physion.analysis import tools


class FaceCameraMixin:

    def has_pupil(self):
        return ('Pupil' in self.nwbfile.processing)

    def has_gaze(self):
        return self.has_pupil()

    def read_pupil(self):
        """
        read metadata
        """

        pd = str(self.nwbfile.processing['Pupil'].description)

        # extract pupil scale
        if len(pd.split('pix_to_mm='))>1:
            self.FaceCamera_mm_to_pix = int(1./float(pd.split('pix_to_mm=')[-1]))
        else:
            self.FaceCamera_mm_to_pix = 1

        # extract pupil ROI
        try:
            self.pupil_ROI = {}
            for key, val in zip(\
                ['xmin','xmax','ymin','ymax'],
                pd.split('pupil ROI: (xmin,xmax,ymin,ymax)=(')[1].split(')')[0].split(',')):
                self.pupil_ROI[key] = int(val)
        except (KeyError, IndexError, ValueError) as be:
            self.pupil_ROI = None

    def build_pupil(self,
                    specific_time_sampling=None,
                    interpolation='linear',
                    verbose=False):
        """
        build pupil diameter trace, i.e. twice the maximum of the ellipse radius at each time point
        """
        if self.has_pupil():

            self.t_pupil = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps[:]
            self.pupil =  2*np.max([self.nwbfile.processing['Pupil'].data_interfaces['sx'].data[:,0],
                                             self.nwbfile.processing['Pupil'].data_interfaces['sy'].data[:,0]], axis=0)

            if verbose:
                print(' [ok] --> "pupil" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_pupil, self.pupil,
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            print(' %s --> pupil diameter not available ...' % self.df_name)

    def build_gaze(self,
                            specific_time_sampling=None,
                            interpolation='linear',
                            verbose=False):
        """
        build gaze movement 

        build distance from mean (x,y) position of pupil
        """
        if self.has_pupil():
            self.t_gaze = self.nwbfile.processing['Pupil'].data_interfaces['cx'].timestamps[:]
            cx = self.nwbfile.processing['Pupil'].data_interfaces['cx'].data[:,0]
            cy = self.nwbfile.processing['Pupil'].data_interfaces['cy'].data[:,0]
            self.gaze = np.sqrt((cx-np.mean(cx))**2+(cy-np.mean(cy))**2)

            if specific_time_sampling is not None:
                return tools.resample(self.t_gaze, self.gaze, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

            if verbose:
                print(' [ok] --> "gaze" built successfully ')

        else:
            print(' %s --> gaze movement not available ...' % self.df_name)

    def has_facemotion(self):
        return ('FaceMotion' in self.nwbfile.processing)

    def read_facemotion(self):
        
        try:
            fd = str(self.nwbfile.processing['FaceMotion'].description)
            self.FaceMotion_ROI = [int(i) for i in fd.split('y0,dy)=(')[1].split(')')[0].split(',')]
        except (KeyError, IndexError, ValueError) as be:
            self.FaceMotion_ROI = None

    def build_facemotion(self,
                         specific_time_sampling=None,
                         interpolation='linear',
                         verbose=False):
        """
        build facemotion
        """

        if self.has_facemotion():

            self.t_facemotion = self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps[:]
            self.facemotion =  self.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[:,0]

            if verbose:
                print(' [ok] --> "facemotion" built successfully ')

            if specific_time_sampling is not None:
                return tools.resample(self.t_facemotion, self.facemotion, 
                                      specific_time_sampling, 
                                      interpolation=interpolation, 
                                      verbose=verbose)

        else:
            print(' %s --> "facemotion" not available ...' % self.df_name)
