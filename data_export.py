import h5py
import numpy as np
import os
import pandas as pd
import argparse

from result import get_all_results

if __name__ == "__main__":
    d = {"dataset": [], "query_set": [], "mu": [], "algorithm": [], "params": [], "rel_err": [], "samples": [], "query_time": [],
         "build_time": []}

    parser = argparse.ArgumentParser()
    parser.add_argument('--mu', default = None, type=float)
    parser.add_argument('--dataset', default = None, type=str)
    parser.add_argument('--algorithm', default = None, type=str)
    parser.add_argument('--sort-by', default = 'query_time', type=str,
                            choices = d.keys())
    parser.add_argument('--query-set', default = None, type=str,
                            choices = ['validation','test'])
    parser.add_argument('--max-err', default = None, type=float)
    parser.add_argument('-o', default = None, type=str, help='output as csv')
    args = parser.parse_args()

    for f in get_all_results():
        algorithm = f.attrs["algorithm"]
        dataset = f.attrs["dataset"]
        mu = f.attrs["mu"]
        query_set = f.attrs['query_set']
        build_time = f.attrs['build_time']

        if args.mu is not None and args.mu != mu:
            continue
        if args.dataset is not None and args.dataset != dataset:
            continue
        if args.algorithm is not None and args.algorithm != algorithm:
            continue
        if args.query_set is not None and args.query_set != query_set:
            continue
        #print(f"Checking {f}")
        with h5py.File(os.path.join("data", f"{f.attrs['dataset']}.hdf5"), 'r') as g:
            try:
                ground_truth_gaussian_str = f'kde.{query_set}.gaussian' + f'{mu:f}'.strip('0')
                assert ground_truth_gaussian_str in g
                ground_truth = np.array(g[ground_truth_gaussian_str])
                # an ugly kludge to remove NaNs
                # ground_truth[ground_truth == 0.0] = np.finfo(np.float32).tiny
                ground_truth[ground_truth < 1e-16] = 1e-16
                err_array = np.abs(np.array(f['estimates']) - ground_truth[:,None])/ground_truth[:,None]

                # this is a very ugly kludge
                # print(err_array)
                # if algorithm.startswith('sklearn') and 'atol=0.0,' in f.attrs['params']:
                #    err_array[err_array > 10] = 0

                rel_err = np.mean(err_array)

                if args.max_err is not None and rel_err > args.max_err:
                    continue

                d['algorithm'].append(algorithm)
                d['dataset'].append(dataset)
                d['query_set'].append(query_set)
                d['mu'].append(mu)
                d['rel_err'].append(rel_err)
                d['samples'].append(np.mean(f['samples']))
                d['query_time'].append(np.mean(f['times']))
                d['params'].append(f.attrs['params'])
                d['build_time'].append(build_time)
            except Exception as e:
                print(f"Couldn't process {f.filename}: {e}")
        f.close()

    df = pd.DataFrame(data=d)
    print(df.sort_values(by=args.sort_by).to_string())

    if args.o is not None:
        df.to_csv(args.o, index = False)
