import numpy as np

def shuffle(protocol, self):
    """
    takes a protocol and a visual_stim object as argument

    performs the shuffling of the array in visual_stim.experiment
    """

    # initialize random seed if provided
    if 'shuffling-seed' in protocol:
        np.random.seed(protocol['shuffling-seed']) 


    

if __name__=='__main__':

    print(3)
