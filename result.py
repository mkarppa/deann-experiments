import os
import h5py
import pandas as pd

def get_result_fn(dataset, mu, query_set, algo, args_str, query_str):
    dir_name = os.path.join("results", dataset, query_set, algo, str(mu))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return os.path.join(dir_name, f"{args_str}_{query_str}.hdf5")

def result_exists(dataset, mu, query_set, algo, args_str, query_str):
    return os.path.exists(get_result_fn(dataset, mu, query_set, 
        algo, args_str, query_str))


def write_result(res, ds, mu, query_set, algo, args_str, query_str, err=None):
    # ids, ests, samples, times = algo.process_result()
    # if len(ids) != m:
    #     print(f"Couldn't fetch results for {algo.name()} running with {str(algo)}.")
    #     return

    fn = get_result_fn(ds, mu, query_set, algo, args_str, query_str)
    with h5py.File(fn, 'w') as f:
        f.attrs['dataset'] = ds
        f.attrs['algorithm'] = algo 
        f.attrs['params'] = args_str + "_" + query_str 
        f.attrs['mu'] = mu
        f.attrs['query_set'] = query_set
        if err:
            f.attrs['err'] = err
        else:
            # f.create_dataset('ids', data=pivot['id'])
            pivot = pd.pivot(res, columns='iter', index='id')
            f.create_dataset('estimates', data=pivot['est'])
            f.create_dataset('samples', data=pivot['samples'])
            f.create_dataset('times', data=pivot['time'])
