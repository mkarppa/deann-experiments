import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from itertools import product
from preprocess_datasets import get_dataset,DATASETS

import argparse
import yaml
import h5py
import numpy as np
import pandas as pd
import time
import json


def get_result_fn(dataset, mu, query_set, algo):
    print(dataset, mu, query_set, algo.name(), algo)
    dir_name = os.path.join("results", dataset, query_set, algo.name(), str(mu))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return os.path.join(dir_name, f"{algo}.hdf5")

def write_result(res, ds, mu, algo, query_set, m):
    # ids, ests, samples, times = algo.process_result()
    # if len(ids) != m:
    #     print(f"Couldn't fetch results for {algo.name()} running with {str(algo)}.")
    #     return

    fn = get_result_fn(ds, mu, query_set, algo)
    pivot = pd.pivot(res, columns='iter', index='id')
    with h5py.File(fn, 'w') as f:
        f.attrs['dataset'] = ds
        f.attrs['algorithm'] = algo.name()
        f.attrs['params'] = str(algo)
        f.attrs['mu'] = mu
        f.attrs['query_set'] = query_set
        # f.create_dataset('ids', data=pivot['id'])
        f.create_dataset('estimates', data=pivot['est'])
        f.create_dataset('samples', data=pivot['samples'])
        f.create_dataset('times', data=pivot['time'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--kde-value',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='shuttle',
    )
    parser.add_argument(
        '--definition',
        default='algos.yaml',
    )
    parser.add_argument(
        '--algorithm',
    )
    parser.add_argument(
        '--force',
        help='overwrite existing results',
        action="store_true"
    )
    parser.add_argument(
        '--list-algorithms',
        action="store_true",
    )
    parser.add_argument(
        '--reps',
        type=int,
        default=1
    )
    parser.add_argument(
        '--query-set',
        choices=['validation','test'],
        default='test'
    )
    args = parser.parse_args()

    with open(args.definition, 'r') as f:
        definitions = yaml.load(f, Loader=yaml.Loader)

    print(definitions)

    if args.list_algorithms:
        print("Available algorithms:")
        print("\n".join(definitions))
        exit(0)

    print(f"Running on {args.dataset}")
    dataset_name = args.dataset
    dataset = get_dataset(dataset_name)
    print(dataset)

    mu = args.kde_value
    kde_str = 'kde.' + args.query_set + '{:f}'.format(mu).strip('0')
    _, bw = dataset.attrs[kde_str]

    print(f"Running with bandwidth {bw} to achieve kde value {mu}.")


    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = list(definitions.keys())

    X = np.array(dataset['train'], dtype=np.float32)
    Y = np.array(dataset[args.query_set], dtype=np.float32)

    tau = np.percentile(np.array(dataset[f'kde.{args.query_set}' + f'{mu:f}'.strip('0')], dtype=np.float32),
                            1)

    exps = {}

    for algo in algorithms:
        mod = __import__(f'algorithms.{definitions[algo]["wrapper"]}', fromlist=[definitions[algo]['constructor']])
        Est_class = getattr(mod, definitions[algo]['constructor'])
        est = Est_class(dataset_name, args.query_set, mu, bw, definitions[algo].get('args', {}))
        for query_params in definitions[algo].get('query', [None]):
            if type(query_params) == list and type(query_params[0]) == list:
                qps = product(*query_params)
            else:
                qps = [query_params]
            for qp in qps: 
                est.set_query_param(qp)
                if args.force or not os.path.exists(get_result_fn(dataset_name, mu, args.query_set, est)):
                    exps.setdefault(algo, []).append(qp)



    print(exps)

# generate all experiments and remove the once that are already there
    for algo, query_params in exps.items():
        mod = __import__(f'algorithms.{definitions[algo]["wrapper"]}', fromlist=[definitions[algo]['constructor']])
        Est_class = getattr(mod, definitions[algo]['constructor'])
        est = Est_class(dataset_name, args.query_set, mu, bw, definitions[algo].get('args', {}))
        print(f'Running {algo}')
        t0 = time.time()
        # est.fit(numpy.array(X, dtype=numpy.float32))
        est.fit(X)
        print(f'Preprocessing took {(time.time() - t0)/1e6} ms.')
        for query_params in query_params:
            print(f'Running {algo} with {query_params}')
            results = list()
            est.set_query_param(query_params)
            # est.query(numpy.array(Y, dtype=numpy.float32))
            for rep in range(args.reps):
                results.append(est.query(Y))
            try:
                processed_results = est.process_results(results)
            # write_result(dataset_name, mu, est, args.query_set, Y.shape[0])
                write_result(processed_results, dataset_name, mu, est, args.query_set, Y.shape[0])
            except:
                print(f"Error processing {algo} with {query_params}")



if __name__ == "__main__":
    main()



