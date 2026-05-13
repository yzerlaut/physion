import pynwb

def add_ephys(nwbfile, args,
              metadata=None):
    """
    See:
    https://pynwb.readthedocs.io/en/dev/tutorials/domain/ecephys.html
    """

    #   load the open-ephys data:
    # - session
    session = OpenEphysSession(args.NPX_folder)
    # - recording node
    node = int(args.NPX_rec.split('node')[1].split('/')[0])
    rec_id = int(args.NPX_rec.split('rec')[1])-1
    rec = session.recordnodes[node].recordings[rec_id]

    #   create the device model 
    device_model = nwbfile.create_device_model(
                        name="Neuropixels 2.0",
                        manufacturer="Array Technologies",
                        # model_number="PRB_1_4_0480_123",
                        description="A 12-channel array with 4 shanks and 3 channels per shank")
    #   create the device 
    device = nwbfile.create_device(
                        name="array",
                        description="A 12-channel array with 4 shanks and 3 channels per shank",
                        serial_number="1234567890",
                        model=device_model,
                    )

    nwbfile.add_electrode_column(name="label", 
                                 description="label of electrode")

    nshanks = 1 # rec.continuous
    nchannels_per_shank = rec.continuous['ProbeA'].samples.shape[1]
    electrode_counter = 0

    for ishank in range(nshanks):
        # create an electrode group for this shank
        electrode_group = nwbfile.create_electrode_group(
            name="shank{}".format(ishank),
            description="electrode group for shank {}".format(ishank),
            device=device,
            location="brain area",
        )
        # add electrodes to the electrode table
        for ielec in range(nchannels_per_shank):
            nwbfile.add_electrode(
                group=electrode_group,
                label="shank{}elec{}".format(ishank, ielec),
                location="brain area",
            )
            electrode_counter += 1

    # from pynwb.ecephys import LFP, ElectricalSeries, SpikeEventSeries
