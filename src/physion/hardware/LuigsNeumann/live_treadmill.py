import numpy as np
import time
import sys
import nidaqmx

# sys.path += ["C:\LuigsNeumann_Treadmill\IO\ReadWriteNI\PyDAQmx-1.2.3"]
# sys.path += ["C:\LuigsNeumann_Treadmill\IO\ReadWriteNI\PyDAQmx-1.2.3\PyDAQmx"]
# sys.path.append('C:\LuigsNeumann_Treadmill\IO\ReadWriteNI')
# import PyDAQmx
# from PyDAQmx import Task
# from PyDAQmx.DAQmxTypes import *
# from PyDAQmx.DAQmxConstants import *
# Dev=sys.argv[1]
# ch=sys.argv[2]



import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
from nidaqmx.constants import AcquisitionType, LineGrouping
from collections import deque
import time

from physion.hardware.NIdaq.config import find_usb_devices, get_digital_input_channels

# find device
device = find_usb_devices()[0]
print(get_digital_input_channels(device))

# Configure NI DAQmx settings
task = nidaqmx.Task()
task.di_channels.add_di_chan("%s/port0" % device.name, line_grouping=LineGrouping.CHAN_FOR_ALL_LINES)
# task.timing.cfg_samp_clk_timing(rate=1000) #, sample_mode=AcquisitionType.CONTINUOUS)

# Create the plot
plt.ion()  # Enable interactive mode for dynamic updating
fig, ax = plt.subplots(figsize=(7,4))
ax.set_xlabel('time (s)')
ax.set_ylabel('position (?)')
ax.set_title('Digital Input from Dev2/port0')

task.start()

speed= [0]
t0 = time.time()
times = [0]
line, = ax.plot(times, speed)
val=0           # new value
val1=0          # value prior to val
val2=0          # value prior to val1
counter = 0
lastwrite = time.time()

while True:
    try:

        val = task.read(number_of_samples_per_channel=1)[0]

        if val!=val1:                           # value has changed
            if val==3:                          # both channels true
                if val2==0 and val1==1:         # both false --> ch1 true --> ch2 true          
                    counter += 1
                elif val2==0 and val1==2:       # both false --> ch2 true --> ch1 true 
                    counter -= 1

            val2=val1                           # update values
            val1=val

        if time.time()-lastwrite>.04:
            lastwrite=time.time() 
            speed.append(counter)                  #   ==> increase counter
            times.append(time.time()-t0)

            line.set_xdata(times[-400:])
            line.set_ydata(speed[-400:])
            ax.relim()
            ax.autoscale_view()
            plt.pause(0.01)  # Pause to allow the plot to update

    except (KeyboardInterrupt,SystemExit):
        task.stop()
        task.close()
        plt.ioff()  # Turn off interactive mode
        break
    


def _calc_positionIncr(ch1,Dev):

    digital_input1=Task()
    read=int32()
    data1=np.zeros((1,),dtype=np.uint8)
    
    Channel1="Dev"+str(Dev)+"/port"+str(ch1)
    digital_input1.CreateDIChan(Channel1,"",DAQmx_Val_ChanForAllLines)

    digital_input1.StartTask()

    digital_input1.ReadDigitalU8(1,1,DAQmx_Val_GroupByChannel,data1,1,byref(read),None)
    data1_Old=data1[0]
    counter=0
    print(counter)
    lastwrite=time.time()

    val=0           # new value
    val1=0          # value prior to val
    val2=0          # value prior to val1

    while True:
        digital_input1.ReadDigitalU8(1,1,DAQmx_Val_GroupByChannel,data1,1,byref(read),None)         # read value of respective port
        val=int(data1%4)                                                                            # use line 0 and 1 (integer modulo 4)
        if val!=val1:                           # value has changed
            if val==3:                          # both channels true
                if val2==0 and val1==1:         # both false --> ch1 true --> ch2 true          
                    counter+=1                  #   ==> increase counter
                elif val2==0 and val1==2:       # both false --> ch2 true --> ch1 true          
                    counter-=1                  #   ==> decrease counter

            val2=val1                           # update values
            val1=val

        if time.time()-lastwrite>.04:
            print(counter)
            #print val,val1,val2
            sys.stdout.flush()
            lastwrite=time.time() 
                    
            
        
#_calc_positionIncr(ch,Dev)
