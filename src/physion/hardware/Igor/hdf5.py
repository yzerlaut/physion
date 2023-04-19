"""
taken from:
http://codereview.stackexchange.com/questions/120802/recursively-save-python-dictionaries-to-hdf5-files-using-h5py?newreg=f582be64155a4c0f989a2aa05ee67efe

Updated July 2018 
the string encoding has changed
so string keys are translated to bytes
the for the decoding, bytes strings are decoded to utf-8 strings
"""

import numpy as np
import h5py
import os


def make_writable_elements(value):
    # all cases to be covered should be:
    # np.ndarray, np.int64, np.float64, bytes, dict, tuple, list, str
    if isinstance(value, (np.int64, np.float64)):
        return value
    elif isinstance(value, float):
        return np.float(value)
    elif isinstance(value, (int, np.int32)):
        return np.int(value)
    elif isinstance(value, str):
        return np.string_(value)
    elif isinstance(value, tuple):
        return np.array(value)
    else:
        return value

def make_writable_list(List):
    list_to_return = []
    try:
        # if list of lists
        if isinstance(List[0], (list, np.ndarray)):
            list_to_return = [make_writable_list(List[i]) for i in range(len(List))]
        else:
            list_to_return = [make_writable_elements(List[i]) for i in range(len(List))]
    except IndexError:
            list_to_return = [make_writable_elements(List[i]) for i in range(len(List))]
    return list_to_return


def make_writable_dict(dic):
    dic2 = dic.copy()
    for key, value in dic.items():
        if isinstance(value, (list, np.ndarray)):
            dic2[key] = make_writable_list(value)
        elif isinstance(value, dict):
            dic2[key] = make_writable_dict(value)
        else:
            dic2[key] = make_writable_elements(value)
    return dic2

def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        new_key = np.string_(path+key)
        if isinstance(item, np.ndarray):
            if item.dtype.char=='U':
                h5file[new_key] = np.array(item, dtype=np.string_)
            else:
                h5file[new_key] = item
        elif isinstance(item, (np.int64, np.float64)):
            h5file[new_key] = item
        elif isinstance(item, (int, np.int32, np.float32, float, str, bytes, tuple)):
            h5file[new_key] = make_writable_elements(item)
        elif isinstance(item, list):
            h5file[new_key] = make_writable_list(item)
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

        
def make_readable_elements(value):
    # all cases to be covered should be:
    # np.ndarray, np.int64, np.float64, bytes, dict, tuple, list, str
    if isinstance(value, bytes):
        return str(value, 'utf-8')
    elif isinstance(value, np.ndarray):
        if value.dtype.char=='S':
            return np.array(value, dtype=str)
        else:
            return value
    else:
        return value

def make_readable_list(List):
    list_to_return = []
    try:
        # if list of lists
        if isinstance(List[0], (list, np.ndarray)):
            list_to_return = [make_readable_list(List[i]) for i in range(len(List))]
        else:
            list_to_return = [make_readable_elements(List[i]) for i in range(len(List))]
    except IndexError:
        list_to_return = [make_readable_elements(List[i]) for i in range(len(List))]
    return list_to_return


def make_readable_dict(dic):
    dic2 = dic.copy()
    for key, value in dic2.items():
        if isinstance(value, dict):
            dic2[key] = make_readable_dict(value)
        else:
            dic2[key] = make_readable_elements(value)
    return dic2

def load_dict_from_hdf5(filename, verbose=False):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/', verbose=verbose)


def recursively_load_dict_contents_from_group(h5file, path, verbose=False):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        try:
            if isinstance(item, h5py._hl.dataset.Dataset):
                if isinstance(item[()], bytes):
                    to_be_put = str(item[()],'utf-8')
                elif isinstance(item[()], list):
                    to_be_put = make_readable_list(item[()])
                elif isinstance(item[()], np.ndarray):
                    to_be_put = item[()]
                else:
                    to_be_put = make_readable_elements(item[()])
                ans[str(key)] = to_be_put
            elif isinstance(item, h5py._hl.group.Group):
                ans[str(key)] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
        except OSError:
            # print('------------------------------------------------')
            # print(item.attrs.values())
            # print(item.attrs.keys())
            if verbose:
                print('The following was not recognized', key, item)
            else:
                pass
    return ans


if __name__ == '__main__':

    # data = {'x': 'astring',
    #         'y': np.arange(10),
    #         '0_params': {'N': 4000, 'b': 0.0, 'Vthre': -50.0, 'name': 'RecExc', 'Cm': 200.0, 'El': -70.0, 'Trefrac': 5.0, 'tauw': 1000000000.0, 'Vreset': -70.0, 'Gl': 10.0, 'a': 0.0, 'delta_v': 0.0},
    #         '1_params': {'N': 1000, 'b': 0.0, 'Cm': 200.0, 'Vthre': -53.0, 'Gl': 10.0, 'El': -70.0, 'Trefrac': 5.0, 'tauw': 1000000000.0, 'name': 'RecInh', 'Vreset': -70.0, 'a': 0.0, 'delta_v': 0.0},
    #         '0':{'asdfsd':34, 'asd':np.ones(2), 'name':'234'},
    #         '1':np.array([['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd'],['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd']], dtype=str),
    #         '2':[['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd'],['asdfsd', 'asdfsd', 'asdfsd', 'asdfsd']],
    #         'd': {'z': np.ones((2,3)),
    #               'sdkfjh':'',
    #               'dict_of_dict':{'234':'kjsdfhsdjfh','z': np.ones((1,3))},
    #               'b': b'bytestring'}}
    # save_dict_to_hdf5(data, filename)
    # dd = load_dict_from_hdf5(filename)
    # print(dd)
    import sys
    filename = sys.argv[-1]
    # f = h5py.File(filename)
    # for key in f.keys():
    #     print(key)
    # item = f['RecordB9']
    # print(isinstance(item, h5py._hl.dataset.Dataset))
    # print(isinstance(item[()], np.ndarray))
    print(filename)
    dd = load_dict_from_hdf5(filename)
    print(dd.keys())
    # for key, val in dd.items():
    #     print(key)
    # print(f['RecordB9'].dtype)

