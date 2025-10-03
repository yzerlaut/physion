import numpy as np

def shuffle_single_protocol(indices, repeats, protocol,
                            default_seed=1):
    """
    takes a protocol and a visual_stim object as argument

    performs the shuffling of the array in visual_stim.experiment
    """

    # initialize random seed if provided
    if 'shuffling-seed' in protocol:
        np.random.seed(protocol['shuffling-seed']) 
    else:
        np.random.seed(default_seed)

    # -----------------------------------------------------------------
    # TO REMOVE WHEN NOT USING "Randomized-Sequence" ANYMORE
    #   backward compatbility, sometimes no "shuffling" key anymore
    if 'shuffling' not in protocol:
        # this means that protocol['Presentation']=='Randomized-Sequence'
        protocol['shuffling'] = 'full'
    # -----------------------------------------------------------------

    full_indices = np.arange(len(indices))

    if (protocol['shuffling']=='full'):

        np.random.shuffle(full_indices)

    elif (protocol['shuffling']==\
            'full-with-alternate-even-odd-repeats'):

        # extracting even and odd full indices
        full_indices_even = full_indices[repeats%2==0]
        full_indices_odd = full_indices[repeats%2==1]

        # shuffle them
        np.random.shuffle(full_indices_even)
        np.random.shuffle(full_indices_odd)

        # refill full_indices by alternating even and odd episodes
        for i in range(int(len(indices)/2)):

            full_indices[2*i] = full_indices_even[i]
            full_indices[2*i+1] = full_indices_odd[i]
         
    else:
        print()
        print("""
            ###############################################
            ###  [!!] shuffling key not recognized [!!] ### 
            ###        ---> data win't be shuffled !!   ### 
            ###############################################
        """)
        print()

    return indices[full_indices], repeats[full_indices]

def shuffle_multiprotocol(indices, repeats, protocol_id,
                          protocol,
                          default_seed=1):
    full_indices = np.arange(len(indices))

    if (protocol['shuffling']=='full'):

        np.random.shuffle(full_indices)

    return indices[full_indices]

    """
    if (protocol['shuffling']=='full'):
        # print('full shuffling of multi-protocol sequence !')
        np.random.seed(protocol['shuffling-seed']) # initializing random seed
        np.random.shuffle(indices)

        for key in self.experiment:
            self.experiment[key] = np.array(self.experiment[key])[indices]

    if (protocol['shuffling']=='full-alternate-even-odd-repeats'):
        # print('full shuffling of multi-protocol sequence !')
        np.random.seed(protocol['shuffling-seed']) # initializing random seed
        indices = np.arange(len(self.experiment['index']))
        np.random.shuffle(indices)

        for key in self.experiment:
            self.experiment[key] = np.array(self.experiment[key])[indices]

    if (protocol['shuffling']=='per-repeat'):
        # TO BE TESTED
        indices = np.arange(len(self.experiment['index']))
        new_indices = []
        for r in np.unique(self.experiment['repeat']):
            repeat_cond = np.argwhere(self.experiment['repeat']==r).flatten()
            r_indices = indices[repeat_cond]
            np.random.shuffle(r_indices)
            new_indices = np.concatenate([new_indices, r_indices])

        for key in self.experiment:
            self.experiment[key] = np.array(self.experiment[key])[new_indices]
    """

if __name__=='__main__':

    import json, sys
    from physion.visual_stim.build import build_stim

    filename = sys.argv[-1]
    if '.json' in filename:
        with open(filename, 'r') as fp:
            protocol = json.load(fp)
        stim = build_stim(protocol)
        print(stim.experiment)
        print(stim.experiment['repeat'])
    else:
        print("""
                need to provide a json file as argument
              """)

