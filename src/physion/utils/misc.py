import tempfile
import numpy as np

class dummy_datafolder:
    def __init__(self):
        pass
    def get(self):
        return tempfile.gettempdir()

class dummy_parent:
    def __init__(self):
        self.stop_flag = False
        self.datafolder = dummy_datafolder()

def parse_int_list(string):
    """
    "9,3,4,"  -> [9, 3, 4]
    "9-13"    -> [9, 10, 11, 12, 13]
    "1,3-5,8" -> [1, 3, 4, 5, 8]
    """
    values = []
    for item in string.replace(' ', '').split(','):
        if item=='':
            continue
        if '-' in item[1:]: # range (a leading "-" is a negative sign)
            i = item.index('-', 1)
            start, stop = int(item[:i]), int(item[i+1:])
            values += list(range(start, stop+1))
        else:
            values.append(int(item))
    return np.array(values, dtype=int)
