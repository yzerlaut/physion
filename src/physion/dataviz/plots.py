import os, sys, pathlib
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui, QtCore

from physion.dataviz.tools import convert_times_to_indices, convert_index_to_time,\
        convert_time_to_index, scale_and_position, settings
from physion.pupil import process


def raw_data_plot(self, tzoom,
                  plot_update=True,
                  with_images=False,
                  with_roi=False,
                  with_scatter=False):

    iplot = 0
    scatter = []
    self.plot.clear()
    
    y = np.zeros(2)
    ## -------- Screen --------- ##

    if 'Photodiode-Signal' in self.data.nwbfile.acquisition and self.photodiodeSelect.isChecked():
        
        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.acquisition['Photodiode-Signal'])
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.data.nwbfile.acquisition['Photodiode-Signal'].data[list(isampling)], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.data.nwbfile.acquisition['Photodiode-Signal']), y,
                       pen=pg.mkPen(color=settings['colors']['Screen']))

    ## -------- Locomotion --------- ##
    
    if 'Running-Speed' in self.data.nwbfile.acquisition and self.runSelect.isChecked():
        
        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.acquisition['Running-Speed'])
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1+1, i2-1)
        else:
            isampling = np.unique(np.linspace(i1+1, i2-1, settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.data.nwbfile.acquisition['Running-Speed'].data[list(isampling)], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.data.nwbfile.acquisition['Running-Speed']), y,
                       pen=pg.mkPen(color=settings['colors']['Locomotion']))
            

    ## -------- FaceCamera, Face motion and Pupil-Size --------- ##
    
    if 'FaceCamera' in self.data.nwbfile.acquisition and self.imgSelect.isChecked():
        
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['FaceCamera'])
        self.pFaceimg.setImage(self.data.nwbfile.acquisition['FaceCamera'].data[i0].T)
        
        if hasattr(self, 'FaceCameraFrameLevel'):
            self.plot.removeItem(self.FaceCameraFrameLevel)
        self.FaceCameraFrameLevel = self.plot.plot(self.data.nwbfile.acquisition['FaceCamera'].timestamps[i0]*np.ones(2),
                                                   [0, y.max()], pen=pg.mkPen(color=settings['colors']['FaceMotion']), linewidth=0.5)


    if 'FaceMotion' in self.data.nwbfile.acquisition and self.imgSelect.isChecked():
        
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['FaceMotion'])
        self.pFacemotionimg.setImage(self.data.nwbfile.acquisition['FaceMotion'].data[i0].T)
        if hasattr(self, 'FacemotionFrameLevel'):
            self.plot.removeItem(self.FacemotionFrameLevel)
        self.FacemotionFrameLevel = self.plot.plot(self.data.nwbfile.acquisition['FaceMotion'].timestamps[i0]*np.ones(2),
                                                   [0, y.max()], pen=pg.mkPen(color=settings['colors']['FaceMotion']), linewidth=0.5)
        t_facemotion_frame = self.data.nwbfile.acquisition['FaceMotion'].timestamps[i0]
        
    else:
        t_facemotion_frame = None


    if 'FaceMotion' in self.data.nwbfile.processing and self.facemotionSelect.isChecked():

        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'])
        t = self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].timestamps[i1:i2]
        y = scale_and_position(self, self.data.nwbfile.processing['FaceMotion'].data_interfaces['face-motion'].data[i1:i2],
                               i=iplot)
        self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['FaceMotion']))

        # adding grooming flag (dots at the bottom)
        if 'grooming' in self.data.nwbfile.processing['FaceMotion'].data_interfaces:
            cond = (self.data.nwbfile.processing['FaceMotion'].data_interfaces['grooming'].data[i1:i2]==1) & np.isfinite(y)
            if np.sum(cond):
                self.plot.plot(t[cond],y[cond].min()+0*t[cond], pen=None, symbol='o',
                               symbolPen=pg.mkPen(color=settings['colors']['FaceMotion'], width=0),                                      
                               symbolBrush=pg.mkBrush(0, 255, 0, 255), symbolSize=7)
                
        iplot+=1
        
        # self.facemotionROI        


    if 'Pupil' in self.data.nwbfile.acquisition and self.imgSelect.isChecked():
        
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['Pupil'])
        img = self.data.nwbfile.acquisition['Pupil'].data[i0].T
        img = (img-img.min())/(img.max()-img.min())
        self.pPupilimg.setImage(255*(1-np.exp(-img/0.2)))
        if hasattr(self, 'PupilFrameLevel'):
            self.plot.removeItem(self.PupilFrameLevel)
        self.PupilFrameLevel = self.plot.plot(self.data.nwbfile.acquisition['Pupil'].timestamps[i0]*np.ones(2),
                                              [0, y.max()], pen=pg.mkPen(color=settings['colors']['Pupil']), linewidth=0.5)
        t_pupil_frame = self.data.nwbfile.acquisition['Pupil'].timestamps[i0]
    else:
        t_pupil_frame = None
        
            
    if 'Pupil' in self.data.nwbfile.processing:

        i1, i2 = convert_times_to_indices(*tzoom, self.data.nwbfile.processing['Pupil'].data_interfaces['cx'])
        t = self.data.nwbfile.processing['Pupil'].data_interfaces['sx'].timestamps[i1:i2]

        
        if self.gazeSelect.isChecked():

            y = scale_and_position(self,
                                   np.sqrt((self.data.nwbfile.processing['Pupil'].data_interfaces['cx'].data[i1:i2]-self.gaze_center[0])**2+\
                                           (self.data.nwbfile.processing['Pupil'].data_interfaces['cy'].data[i1:i2]-self.gaze_center[1])**2),
                                   i=iplot)
            self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Gaze']))
            
            iplot+=1
            
        if self.pupilSelect.isChecked():
            
            y = scale_and_position(self,
                                   np.max([self.data.nwbfile.processing['Pupil'].data_interfaces['sx'].data[i1:i2],
                                           self.data.nwbfile.processing['Pupil'].data_interfaces['sy'].data[i1:i2]], axis=0),
                                   i=iplot)
            self.plot.plot(t, y, pen=pg.mkPen(color=settings['colors']['Pupil']))

            # adding blinking flag (dots at the bottom)
            if 'blinking' in self.data.nwbfile.processing['Pupil'].data_interfaces:
                cond = (self.data.nwbfile.processing['Pupil'].data_interfaces['blinking'].data[i1:i2]==1) & np.isfinite(y)
                if np.sum(cond):
                    self.plot.plot(t[cond],y[cond].min()+0*t[cond], pen=None, symbol='o',
                                   symbolPen=pg.mkPen(color=settings['colors']['Pupil'], width=0),                                      
                                   symbolBrush=pg.mkBrush(0, 0, 255, 255), symbolSize=7)

            iplot+=1

        # plotting a circle for the pupil fit
        coords = []
        if t_pupil_frame is not None:
            i0 = convert_time_to_index(t_pupil_frame, self.data.nwbfile.processing['Pupil'].data_interfaces['sx'])
            for key in ['cx', 'cy', 'sx', 'sy']:
                coords.append(self.data.nwbfile.processing['Pupil'].data_interfaces[key].data[i0]*self.pupil_mm_to_pix)
            if 'angle' in self.data.nwbfile.processing['Pupil'].data_interfaces:
                coords.append(self.data.nwbfile.processing['Pupil'].data_interfaces['angle'].data[i0])
            else:
                coords.append(0)

            self.pupilContour.setData(*process.ellipse_coords(*coords, transpose=True), size=3, brush=pg.mkBrush(255,0,0))
            

    # ## -------- Electrophy --------- ##
    
    if ('Electrophysiological-Signal' in self.data.nwbfile.acquisition) and self.ephysSelect.isChecked():
        # deprecated
        
        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.acquisition['Electrophysiological-Signal'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.acquisition['Electrophysiological-Signal'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.data.nwbfile.acquisition['Electrophysiological-Signal'].data[list(isampling)], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.data.nwbfile.acquisition['Electrophysiological-Signal']), y,
                       pen=pg.mkPen(color=settings['colors']['Electrophy']))

    if ('LFP' in self.data.nwbfile.acquisition) and self.ephysSelect.isChecked():
        
        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.acquisition['LFP'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.acquisition['LFP'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.data.nwbfile.acquisition['LFP'].data[list(isampling)], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.data.nwbfile.acquisition['LFP']), y,
                       pen=pg.mkPen(color=settings['colors']['LFP']))


    if ('Vm' in self.data.nwbfile.acquisition) and self.ephysSelect.isChecked():
        
        i1 = convert_time_to_index(tzoom[0], self.data.nwbfile.acquisition['Vm'])+1
        i2 = convert_time_to_index(tzoom[1], self.data.nwbfile.acquisition['Vm'])-1
        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))
        y = scale_and_position(self,self.data.nwbfile.acquisition['Vm'].data[list(isampling)], i=iplot)
        iplot+=1
        self.plot.plot(convert_index_to_time(isampling, self.data.nwbfile.acquisition['Vm']), y,
                       pen=pg.mkPen(color=settings['colors']['Vm']))

        
    # ## -------- Calcium --------- ##
    
    if hasattr(self, 'roiIndices'):
        roiIndices = self.roiIndices
    else:
        roiIndices = np.random.choice(np.arange(self.data.nROIs), 10)

    # if (self.time==0) and ('ophys' in self.data.nwbfile.processing):
    if ('ophys' in self.data.nwbfile.processing):

        self.pCaimg.setImage(self.data.nwbfile.processing['ophys'].data_interfaces['Backgrounds_0'].images['meanImg'][:]**.25) # plotting the mean image
        
    if 'CaImaging-TimeSeries' in self.data.nwbfile.acquisition and self.ophysSelect.isChecked():
        i0 = convert_time_to_index(self.time, self.data.nwbfile.acquisition['CaImaging-TimeSeries'])
        # self.pCaimg.setImage(self.data.nwbfile.acquisition['CaImaging-TimeSeries'].data[i0,:,:]) # REMOVE NOW, MAYBE REINTRODUCE
        if hasattr(self, 'CaFrameLevel'):
            self.plot.removeItem(self.CaFrameLevel)
        self.CaFrameLevel = self.plot.plot(self.data.nwbfile.acquisition['CaImaging-TimeSeries'].timestamps[i0]*np.ones(2), [0, y.max()],
                                           pen=pg.mkPen(color=settings['colors']['CaImaging']), linewidth=0.5)
        
    if ('ophys' in self.data.nwbfile.processing) and with_roi:
        if hasattr(self, 'ROIscatter'):
            self.pCa.removeItem(self.ROIscatter)
        self.ROIscatter = pg.ScatterPlotItem()
        X, Y = [], []
        for ir in self.data.valid_roiIndices[roiIndices]:
            indices = np.arange((self.data.pixel_masks_index[ir-1] if ir>0 else 0),
                                (self.data.pixel_masks_index[ir] if ir<len(self.data.valid_roiIndices) else len(self.data.pixel_masks_index)))
            x = [self.data.pixel_masks[ii][1] for ii in indices]
            y = [self.data.pixel_masks[ii][0] for ii in indices]
            X += list(np.mean(x)+3*np.std(x)*np.cos(np.linspace(0, 2*np.pi))) # TO PLOT CIRCLES
            Y += list(np.mean(y)+3*np.std(y)*np.sin(np.linspace(0, 2*np.pi)))
            # X += x # TO PLOT THE REAL ROIS
            # Y += y
        self.ROIscatter.setData(X, Y, size=1, brush=pg.mkBrush(0,255,0))
        self.pCa.addItem(self.ROIscatter)

    if ('ophys' in self.data.nwbfile.processing) and (roiIndices is not None) and self.ophysSelect.isChecked():
        if not hasattr(self.data, 'rawFluo'):
            self.data.build_rawFluo()
        i1 = convert_time_to_index(tzoom[0], self.data.Neuropil, axis=1)
        i2 = convert_time_to_index(tzoom[1], self.data.Neuropil, axis=1)

        if not self.sbsmplSelect.isChecked():
            isampling = np.arange(i1,i2)
        else:
            isampling = np.unique(np.linspace(i1, i2, settings['Npoints'], dtype=int))

        tt = np.array(self.data.Neuropil.timestamps[:])[isampling]

        # if self.roiPick.text()=='sum':
        if False:
            y = scale_and_position(self, self.data.rawFluo[:,isampling].mean(axis=0), i=iplot) # valid ROIs inside
            self.plot.plot(tt, y, pen=pg.mkPen(color=(0,250,0), linewidth=1))
        else:
            y = scale_and_position(self, np.arange(2), i=iplot)
            width = y[1]-y[0]
            for n, ir in enumerate(roiIndices):
                F= self.data.rawFluo[ir,isampling] 
                loc = y[0]+n*width/len(roiIndices)
                self.plot.plot(tt, loc+width*(F-F.min())/(F.max()-F.min())/len(roiIndices), pen=pg.mkPen(color=(0, 100, 0)), linewidth=1)
        iplot += 1

    # ## -------- Visual Stimulation --------- ##

    if self.visualStimSelect.isChecked() and ('time_start_realigned' in self.data.nwbfile.stimulus):

        icond = np.argwhere((self.data.nwbfile.stimulus['time_start_realigned'].data[:]<=self.time) & \
                            (self.data.nwbfile.stimulus['time_stop_realigned'].data[:]>=self.time)).flatten()
        try:
            if len(icond)>1:
                    self.pScreenimg.setImage(255*self.data.visual_stim.get_image(icond[0],
                                                                                 self.time-self.data.nwbfile.stimulus['time_start_realigned'].data[icond[0]]))
            elif self.time<=self.data.nwbfile.stimulus['time_start_realigned'].data[0]: # PRE-STIM
                self.pScreenimg.setImage(255*self.data.visual_stim.get_prestim_image())
            elif self.time>=self.data.nwbfile.stimulus['time_stop_realigned'].data[-1]: # POST-STIM
                self.pScreenimg.setImage(255*self.data.visual_stim.get_poststim_image())
            else: # INTER-STIM
                self.pScreenimg.setImage(255*self.data.visual_stim.get_interstim_image())
        except BaseException as be:
            print(be)
            print('pb with image')
            
        self.pScreenimg.setLevels([0,255])

    if self.visualStimSelect.isChecked() and ('time_start_realigned' in self.data.nwbfile.stimulus) and ('time_stop_realigned' in self.data.nwbfile.stimulus):
        # if visual-stim we highlight the stim periods
        icond = np.argwhere((self.data.nwbfile.stimulus['time_start_realigned'].data[:]>tzoom[0]-10) & \
                            (self.data.nwbfile.stimulus['time_stop_realigned'].data[:]<tzoom[1]+10)).flatten()

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
                
                t0 = self.data.nwbfile.stimulus['time_start_realigned'].data[i]
                t1 = self.data.nwbfile.stimulus['time_stop_realigned'].data[i]

                # stimulus area shaded
                self.StimFill.append(self.plot.plot([t0, t1], [0, 0],
                                fillLevel=y.max(), brush=(150,150,150,80)))

                # adding annotation for that episode
                if self.annotSelect.isChecked():
                    self.StimAnnots.append(pg.TextItem())
                    text = 'stim.#%i\n\n' % (i+1)
                    for key in self.data.nwbfile.stimulus.keys(): # 666 means None
                        if (key not in ['time_start', 'time_start_realigned',
                               'time_stop', 'time_stop_realigned', 'protocol-name']) and \
                                       (self.data.nwbfile.stimulus[key].data[i]!=666):
                            text+='%s : %s\n' % (key, str(self.data.nwbfile.stimulus[key].data[i]))
                    self.StimAnnots[-1].setPlainText(text)                    
                    self.StimAnnots[-1].setPos(t0, 0.95*y.max())
                    self.plot.addItem(self.StimAnnots[-1])
                    
    self.plot.setRange(xRange=tzoom, yRange=[0,y.max()], padding=0.0)
    self.frameSlider.setValue(int(self.SliderResolution*(self.time-tzoom[0])/(tzoom[1]-tzoom[0])))
    
    self.plot.show()
