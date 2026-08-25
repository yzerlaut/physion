import os, sys, pathlib
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore
from scipy.ndimage import gaussian_filter1d

from physion.dataviz.tools import convert_times_to_indices, convert_index_to_time,\
        convert_time_to_index, scale_and_position, settings
from physion.pupil import process

def raw_data_plot(self, tzoom):

    self.iplot = 0
    self.plot.clear()
    
    y = np.zeros(2)

    ## -------- Screen --------- ##

    if 'Photodiode-Signal' in self.data.nwbfile.acquisition and self.photodiodeSelect.isChecked():
        
        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.acquisition['Photodiode-Signal'])

        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        t = convert_index_to_time(isampling, self.data.nwbfile.acquisition['Photodiode-Signal'])
        y = scale_and_position(self,self.data.nwbfile.acquisition['Photodiode-Signal'].data[list(isampling),0])
        self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Screen']))

    ## -------- Locomotion --------- ##
    
    if 'Running-Speed' in self.data.nwbfile.acquisition and self.runSelect.isChecked():
        
        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.acquisition['Running-Speed'])

        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1+1, i2-1)
        else:
            isampling = np.unique(np.linspace(i1+1, i2-1, settings['Npoints'], dtype=int))

        t = convert_index_to_time(isampling, self.data.nwbfile.acquisition['Running-Speed'])
        y = scale_and_position(self,self.data.nwbfile.acquisition['Running-Speed'].data[list(isampling),0])
        self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Locomotion']))
            

    ## -------- FaceCamera, Face motion and Pupil-Size --------- ##
    

    if 'FaceMotion' in self.data.nwbfile.acquisition:
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['FaceMotion'])
        t_facemotion_frame = self.data.nwbfile.acquisition['FaceMotion'].timestamps[i0]
    else:
        t_facemotion_frame = None


    if 'FaceMotion' in self.data.nwbfile.processing and self.whiskSelect.isChecked():

        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])
        t = self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps[i1:i2]
        y = scale_and_position(self, self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[i1:i2,0])
        self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['FaceMotion']))

        # adding grooming flag (dots at the bottom)
        if 'grooming' in self.data.nwbfile.processing['FaceMotion'].data_interfaces:
            cond = (self.data.nwbfile.processing['FaceMotion'].data_interfaces['grooming'].data[i1:i2,0]==1) & np.isfinite(y)
            if np.sum(cond):
                self.plot.plot(t[cond],y[cond].min()+0*t[cond], pen=None, symbol='o',
                               symbolPen=pg.mkPen(color=settings['colors']['FaceMotion'], width=0),                                      
                               symbolBrush=pg.mkBrush(0, 255, 0, 255), symbolSize=7)
                


    if 'Pupil' in self.data.nwbfile.acquisition:
        
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['Pupil'])
        t_pupil_frame = self.data.nwbfile.acquisition['Pupil'].timestamps[i0]
        # img = self.data.nwbfile.acquisition['Pupil'].data[i0].T
        # img = (img-img.min())/(img.max()-img.min())
        # self.pPupilimg.setImage(255*(1-np.exp(-img/0.2)))
        # if hasattr(self, 'PupilFrameLevel'):
            # self.plot.removeItem(self.PupilFrameLevel)
        # self.PupilFrameLevel = self.plot.plot(self.data.nwbfile.acquisition['Pupil'].timestamps[i0]*np.ones(2),
                                              # [0, y.max()], pen=pg.mkPen(color=settings['colors']['Pupil']), linewidth=0.5)
    else:
        t_pupil_frame = None
        
            
    if 'Pupil' in self.data.nwbfile.processing:

        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.processing['Pupil'].data_interfaces['cx'])
        t = self.data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[i1:i2]
        
        if self.gazeSelect.isChecked():

            y = scale_and_position(self,
                        np.sqrt((self.data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[i1:i2,0]-self.gaze_center[0])**2+\
                                (self.data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[i1:i2,0]-self.gaze_center[1])**2))
            self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Gaze']))
            
        if self.pupilSelect.isChecked():
            
            y = scale_and_position(self,
                  self.data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[i1:i2,0]*\
                   self.data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[i1:i2,0])

            self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Pupil']))

            # adding blinking flag (dots at the bottom)
            if 'blinking' in self.data.nwbfile.processing['Pupil'].data_interfaces:
                cond = (self.data.nwbfile.processing['Pupil'].data_interfaces['blinking'].data[i1:i2,0]==1) & np.isfinite(y)
                if np.sum(cond):
                    self.plot.plot(t[cond],y[cond].min()+0*t[cond], pen=None, symbol='o',
                                   symbolPen=pg.mkPen(color=settings['colors']['Pupil'], width=0),                                      
                                   symbolBrush=pg.mkBrush(0, 0, 255, 255), symbolSize=7)

            
        
    # ## -------------------------- ## 
    # ## --------   LFP   --------- ##
    # ## -------------------------- ## 

    if ('LFP' in self.data.nwbfile.processing) and\
                            (self.LFPSelect.isChecked()):

        iHeight = 3 # default height of the LFP plot

        try:
            nTraces = int(str(self.LFPSettings.text()).split('n:')[1].split(',')[0])
            smoothing = int(str(self.LFPSettings.text()).split('s:')[1].split(',')[0])
        except BaseException as be:
            print(be)
            print(' LFP options not recognized ! setting defaults ')
            nTraces, smoothing = 4, 10

        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.processing['LFP'].data_interfaces['LFP'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.processing['LFP'].data_interfaces['LFP'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        y = scale_and_position(self, np.arange(2), iHeight=iHeight)
        width = (y[1]-y[0])*.95

        elecs = self.data.nwbfile.processing['LFP'].data_interfaces['LFP'].electrodes
        nElec = len(elecs.data[:])
        elecRange = np.linspace(0, nElec, nTraces+1, dtype=int)

        tt = convert_index_to_time(isampling, self.data.nwbfile.processing['LFP'].data_interfaces['LFP'])

        for n, (e0, e1) in enumerate(zip(elecRange[:-1], elecRange[1:])):

            rdm = np.random.randint(0, 255) # for color
            loc = y[0]+n*width/nTraces
            V = gaussian_filter1d(\
                self.data.nwbfile.processing['LFP'].data_interfaces['LFP'].data[:,e0:e1].mean(axis=-1)[isampling],
                smoothing+1e-6)
            self.plot.plot(tt,
                    loc+1.3*width*(V-V.min())/(V.max()-V.min())/nTraces,
                    pen=pg.mkPen(color=(rdm, 255, 255, 255)))
            
            # roi number annotation
            roiAnnot = pg.TextItem('%i-' % elecs.data[e0],
                                   color=(rdm, 255, 255))
            roiAnnot.setPos(tt[0], loc+width/nTraces/2.)
            self.plot.addItem(roiAnnot)

    # ## -------------------------- ## 
    # ## --------   MUA   --------- ##
    # ## -------------------------- ## 

    if ('MUA' in self.data.nwbfile.processing) and\
                            (self.MUASelect.isChecked()):

        iHeight = 2 # default height of the MUA plot

        try:
            nTraces = int(str(self.MUASettings.text()).split('n:')[1].split(',')[0])
            smoothing = int(str(self.MUASettings.text()).split('s:')[1].split(',')[0])
        except BaseException as be:
            print(be)
            print(' MUA options not recognized ! setting defaults ')
            nTraces, smoothing = 4, 10

        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.processing['MUA'].data_interfaces['MUA'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.processing['MUA'].data_interfaces['MUA'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        y = scale_and_position(self, np.arange(2), iHeight=iHeight)
        width = (y[1]-y[0])*.95

        elecs = self.data.nwbfile.processing['MUA'].data_interfaces['MUA'].electrodes
        nElec = len(elecs.data[:])
        elecRange = np.linspace(0, nElec, nTraces+1, dtype=int)

        tt = convert_index_to_time(isampling, self.data.nwbfile.processing['MUA'].data_interfaces['MUA'])

        for n, (e0, e1) in enumerate(zip(elecRange[:-1], elecRange[1:])):

            rdm = np.random.randint(0, 255) # for color
            loc = y[0]+n*width/nTraces
            V = gaussian_filter1d(\
                self.data.nwbfile.processing['MUA'].data_interfaces['MUA'].data[:,e0:e1].mean(axis=-1)[isampling],
                smoothing+1e-6)
            self.plot.plot(tt,
                    loc+1.3*width*(V-V.min())/(V.max()-V.min())/nTraces,
                    pen=pg.mkPen(color=(rdm, 255, rdm, 255)))
            
            # roi number annotation
            roiAnnot = pg.TextItem('%i-' % elecs.data[e0],
                                   color=(rdm, 255, rdm))
            roiAnnot.setPos(tt[0], loc+width/nTraces/2.)
            self.plot.addItem(roiAnnot)

    # ## -------------------------- ## 
    # ## --------  spikes --------- ##
    # ## -------------------------- ## 

    if ('Spiking' in self.data.nwbfile.processing) and\
                            (self.spikesSelect.isChecked()):

        iHeight = 1 # default height of the spikes plot

        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.processing['MUA'].data_interfaces['MUA'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.processing['MUA'].data_interfaces['MUA'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        y = scale_and_position(self, np.arange(2), iHeight=iHeight)
        width = (y[1]-y[0])*.95

        subsampling = 1 # by default
        if self.sbsmplSelect.isChecked():
            subsampling = 20

        for n in np.arange(len(self.data.nwbfile.units))[::subsampling]:

            spk_times = self.data.nwbfile.units[n].spike_times[n][::subsampling]

            cond = (spk_times>tzoom[0]) & (spk_times<tzoom[1])

            loc = y[0]+n*width/len(self.data.nwbfile.units)

            self.plot.plot(spk_times[cond],
                           loc+np.zeros(len(spk_times[cond])),
                           pen=None, symbol='o', symbolSize=2, symbolPen='w', symbolBrush='w')

    # ## -------------------------- ## 
    # ## -------- Calcium --------- ##
    # ## -------------------------- ## 

    if ('ophys' in self.data.nwbfile.processing) and\
            (self.rawFluoSelect.isChecked() or self.neuropilSelect.isChecked()):

        iHeight = 5 # default height of the CaImaging plot

        try:
            iStart = int(str(self.rawFluoSettings.text()).split('i:')[1].split(',')[0])
            nROIs = int(str(self.rawFluoSettings.text()).split('n:')[1].split(',')[0])
            smoothing = float(str(self.rawFluoSettings.text()).split('s:')[1].split(',')[0])
        except BaseException as be:
            print(be)
            print(' ophys options not recognized ! setting defaults ')
            iStart, nROIs, smoothing = -1, 10, 0

        if iStart==-1:
            # random pick
            roiIndices = np.sort(\
                    np.random.choice(np.arange(self.data.nROIs),
                                          np.min([nROIs, self.data.nROIs]),
                                     replace=False))[::-1]
        else:
            # ordered
            roiIndices = np.arange(iStart,
                            np.min([iStart+nROIs, self.data.nROIs]))[::-1]

        i1 = convert_time_to_index(tzoom[0], self.data.Neuropil, axis=1)
        i2 = convert_time_to_index(tzoom[1], self.data.Neuropil, axis=1)

        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        tt = np.array(self.data.Neuropil.timestamps[:])[isampling]

        y = scale_and_position(self, np.arange(2), iHeight=iHeight)
        width = (y[1]-y[0])

        for n, ir in enumerate(roiIndices):

            loc = y[0]+n*width/len(roiIndices)

            F = gaussian_filter1d(self.data.Fluorescence.data[isampling,ir], smoothing+1e-6)

            if self.neuropilSelect.isChecked():
                Fneu = gaussian_filter1d(self.data.Neuropil.data[isampling,ir], smoothing+1e-6)
                self.plot.plot(tt,
                        loc+1.3*width*(Fneu-Fneu.min())/(Fneu.max()-Fneu.min())/len(roiIndices),
                        pen=pg.mkPen(color=settings['colors']['neuropil']), linewidth=1)

            self.plot.plot(tt,
                    loc+1.3*width*(F-F.min())/(F.max()-F.min())/len(roiIndices),
                    pen=pg.mkPen(color=settings['colors']['rawFluo']), linewidth=1)
            
            # roi number annotation
            roiAnnot = pg.TextItem(str(ir), color=(200, 250, 200))
            roiAnnot.setPos(tt[0], loc+width/len(roiIndices)/2.)
            self.plot.addItem(roiAnnot)


    # ## ------------------------------------- ##
    # ## -------- Visual Stimulation --------- ##
    # ## ------------------------------------- ##

    if self.visualStimSelect.isChecked() and ('time_start_realigned' in self.data.nwbfile.stimulus):

        icond = np.argwhere((self.data.nwbfile.stimulus['time_start_realigned'].data[:,0]<=self.time) & \
                            (self.data.nwbfile.stimulus['time_stop_realigned'].data[:,0]>=self.time)).flatten()

    if self.visualStimSelect.isChecked() and\
            ('time_start_realigned' in self.data.nwbfile.stimulus) and\
            ('time_stop_realigned' in self.data.nwbfile.stimulus):
        # if visual-stim we highlight the stim periods
        icond = np.argwhere((self.data.nwbfile.stimulus['time_start_realigned'].data[:,0]>tzoom[0]-10) & \
                            (self.data.nwbfile.stimulus['time_stop_realigned'].data[:,0]<tzoom[1]+10)).flatten()

        if hasattr(self, 'StimFill') and (self.StimFill is not None):
            for x in self.StimFill:
                self.plot.removeItem(x)
        if hasattr(self, 'StimAnnots') and (self.StimAnnots is not None):
            for x in self.StimAnnots:
                self.plot.removeItem(x)

        X, Y = [], []
        if len(icond)>0:
            
            self.StimFill, self.StimAnnots = [], []

            # looping over episodes
            for i in range(max([0,icond[0]-1]),
                           min([icond[-1]+1,self.data.nwbfile.stimulus['time_stop_realigned'].data.shape[0]])):
                
                t0 = self.data.nwbfile.stimulus['time_start_realigned'].data[i,0]
                t1 = self.data.nwbfile.stimulus['time_stop_realigned'].data[i,0]

                # stimulus area shaded
                self.StimFill.append(self.plot.plot([t0, t1], [0, 0],
                                fillLevel=y.max(), brush=(150,150,150,80)))

                # adding annotation for that episode
                if self.annotSelect.isChecked():
                    self.StimAnnots.append(pg.TextItem())
                    text = 'stim.#%i\n\n' % (i+1)
                    for key in self.data.nwbfile.stimulus.keys(): # 666 means None
                        
                        if key not in ['time_start', 'time_start_realigned',
                                       'time_stop','time_stop_realigned',
                                       'protocol-name']:
                       
                            # handle both 1D and 2D datasets. Is it always 1D now?
                            value = self.data.nwbfile.stimulus[key].data[i] \
                                                if self.data.nwbfile.stimulus[key].data.ndim == 1 \
                                                else self.data.nwbfile.stimulus[key].data[i, 0]

                            if value != 666:
                                text += '%s : %s\n' % (key, str(value))


                    if 'protocol_id' in self.data.nwbfile.stimulus:
                        # handle both 1D and 2D datasets. Is it always 1D now?
                        value_prot = self.data.nwbfile.stimulus['protocol_id'].data[i] \
                                            if self.data.nwbfile.stimulus['protocol_id'].data.ndim == 1 \
                                            else self.data.nwbfile.stimulus['protocol_id'].data[i, 0]
                        text += '\n* %s *\n' % self.data.protocols[value_prot][:20]
                    self.StimAnnots[-1].setPlainText(text)                    
                    self.StimAnnots[-1].setPos(t0, 0.95*y.max())
                    self.plot.addItem(self.StimAnnots[-1])
                    
    self.plot.setRange(xRange=tzoom, yRange=[0,y.max()], padding=0.0)
    self.frameSlider.setValue(int(self.SliderResolution*(self.time-tzoom[0])/(tzoom[1]-tzoom[0])))
    
    self.plot.show()

if __name__=='__main__':

    print('test here')

