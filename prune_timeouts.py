#!/usr/bin/env python3

import os
import h5py

if __name__ == '__main__':
    print('Pruning timeouted results...')
    delete_list = list()
    for root, dirs, files in os.walk('results'):
        for fn in files:
            if fn.endswith('.hdf5'):
                path = root + '/' + fn
                with h5py.File(path, 'r') as f:
                    if 'err' in f.attrs:
                        delete_list.append(path)
    for path in delete_list:
        print(path)
