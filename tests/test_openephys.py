"""
configuration of the Spike Detector of Open Ephys
    (the HTTP server of the GUI is replaced by a fake one)
"""
import xml.etree.ElementTree as ET

import pytest

from physion.ephys import openephys

CONFIG = """<?xml version="1.0" encoding="UTF-8"?>
<SETTINGS>
  <SIGNALCHAIN>
    <PROCESSOR name="OneBox" nodeId="100">
      <CUSTOM_PARAMETERS><KEEP value="1"/></CUSTOM_PARAMETERS>
    </PROCESSOR>
    <PROCESSOR name="Spike Detector" nodeId="106">
      <CUSTOM_PARAMETERS>
        <SPIKE_CHANNEL name="Tetrode 1" num_channels="4" local_channels="2,3,4,5"/>
        <SPIKE_CHANNEL name="Electrode 1" num_channels="1" stream_name="OneBox-ADC"/>
      </CUSTOM_PARAMETERS>
      <EDITOR displayName="Spike Detector"/>
    </PROCESSOR>
  </SIGNALCHAIN>
</SETTINGS>
"""
PROBE = {'name':'ProbeA', 'channel_count':384,
         'sample_rate':30000.0, 'source_id':100}
ADC = {'name':'OneBox-ADC', 'channel_count':12,
       'sample_rate':30300.5, 'source_id':100}
PROCESSORS = [{'id':100, 'name':'OneBox', 'streams':[PROBE, ADC]},
              {'id':106, 'name':'Spike Detector', 'streams':[PROBE, ADC]}]


@pytest.fixture
def gui(monkeypatch, tmp_path):
    """ a fake GUI: "load" makes the loaded file the current config """
    state = {'mode':'IDLE', 'config':CONFIG, 'loaded':[]}

    def get(endpoint):
        return {'/status':{'mode':state['mode']},
                '/processors':{'processors':PROCESSORS},
                '/config':{'info':state['config']}}[endpoint]

    def load(path):
        state['loaded'].append(path)
        with open(path, encoding='utf-8') as f:
            state['config'] = f.read()

    monkeypatch.setattr(openephys, 'get', get)
    monkeypatch.setattr(openephys, 'load_signal_chain', load)
    monkeypatch.setattr(openephys, 'OUTPUT_FOLDER', str(tmp_path))
    return state


def electrodes(config):
    settings = ET.fromstring(config)
    return openephys.find_processor_settings(settings, 106).findall(\
                                        'CUSTOM_PARAMETERS/SPIKE_CHANNEL')


def test_one_electrode_per_contact(gui):
    assert openephys.main(['SPIKE-DETECTOR', '--threshold_type', 'std',
                           '--threshold_value', '4.5'])==0
    channels = electrodes(gui['config'])
    assert len(channels)==384 # previous ones removed
    assert [c.get('local_channels') for c in channels]==\
                                    [str(i+1) for i in range(384)]
    assert channels[0].attrib=={'name':'Electrode 1',
                                'description':'Electrode from Spike Detector 106',
                                'num_channels':'1',
                                'sample_rate':'30000.0',
                                'stream_name':'ProbeA',
                                'stream_source':'100',
                                'local_channels':'1',
                                'thrshlder_type':'1',
                                'abs_threshold1':'-50.0',
                                'std_threshold1':'4.5',
                                'dyn_threshold1':'4.0',
                                'waveform_type':'0'}


def test_abs_threshold(gui):
    assert openephys.main(['SPIKE-DETECTOR', '--threshold_type', 'abs',
                           '--threshold_value', '-60'])==0
    channel = electrodes(gui['config'])[-1]
    assert channel.get('thrshlder_type')=='0'
    assert channel.get('abs_threshold1')=='-60.0'
    assert channel.get('std_threshold1')=='4.0'


def test_other_settings_are_kept(gui):
    assert openephys.main(['SPIKE-DETECTOR'])==0
    settings = ET.fromstring(gui['config'])
    assert settings.find('.//KEEP') is not None
    assert settings.find('.//EDITOR').get('displayName')=='Spike Detector'


@pytest.mark.parametrize('kind, value', [('std', 0.5), ('std', 11),
                                         ('abs', 50), ('dyn', 11)])
def test_threshold_out_of_range(gui, kind, value):
    assert openephys.main(['SPIKE-DETECTOR', '--threshold_type', kind,
                           '--threshold_value=%g' % value])==1
    assert gui['loaded']==[]


@pytest.mark.parametrize('mode', ['ACQUIRE', 'RECORD'])
def test_refused_during_acquisition(gui, mode):
    gui['mode'] = mode
    assert openephys.main(['SPIKE-DETECTOR'])==1
    assert gui['loaded']==[]


def test_unknown_stream(gui):
    assert openephys.main(['SPIKE-DETECTOR', '--stream', 'ProbeB'])==1
    assert gui['loaded']==[]


def test_dry_run_does_not_load(gui):
    gui['mode'] = 'ACQUIRE'
    assert openephys.main(['SPIKE-DETECTOR', '--dry_run'])==0
    assert gui['loaded']==[]


def test_settings_not_applied_by_the_gui(gui, monkeypatch):
    monkeypatch.setattr(openephys, 'load_signal_chain', lambda path: None)
    assert openephys.main(['SPIKE-DETECTOR'])==1
