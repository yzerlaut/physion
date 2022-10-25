import numpy as np

######################################
#### saving and loading NPZ files ####
######################################

def save_dict(filename, data):

    if '.npz' not in filename:
        print('/!\ The filename need to have the "npz" extension')
        print('        ------- renamed to:', filename+'.npz')
        np.savez(filename+'.npz', **data)
    else:
        np.savez(filename, **data)

def load_dict(filename):

    data = np.load(filename, allow_pickle=True)
    output = {}
    for key in data.files:
        output[key] = data[key]

        if type(output[key]) is np.ndarray:
            try:
                output[key] = output[key].item()
            except ValueError:
                pass
        
    return output


