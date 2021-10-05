import argparse 		
import json
import os
import yaml
import h5py
import numpy as np
import time


from itertools import product
from preprocess_datasets import (get_dataset,DATASETS)
from result import (get_result_fn, write_result)

def run_from_cmdline(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mu',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--bw',
        default=0.01,
        type=float
    )
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        default='shuttle',
    )
    parser.add_argument(
        '--wrapper',
    )
    parser.add_argument(
        '--constructor',
    )
    parser.add_argument(
        '--force',
        help='overwrite existing results',
        action="store_true"
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
    parser.add_argument(
        '--query-args',
    )
    parser.add_argument(
        '--build-args',
    )
    parser.add_argument(
        '--algorithm',
    )
    parser.add_argument(
        '--kernel',
    )
    args = parser.parse_args(args)

    query_args = json.loads(args.query_args)
    build_args = json.loads(args.build_args)

    dataset = get_dataset(args.dataset, args.kernel)
    algo = args.algorithm

    X = np.array(dataset['train'], dtype=np.float32)
    Y = np.array(dataset[args.query_set], dtype=np.float32)

    print(query_args)

    mod = __import__(f'algorithms.{args.wrapper}', fromlist=[args.constructor])
    Est_class = getattr(mod, args.constructor)
    est = Est_class(args.dataset, args.query_set, args.kernel, args.mu, args.bw, build_args)
    print(f'Running {algo}')
    t0 = time.time()
    # est.fit(numpy.array(X, dtype=numpy.float32))
    est.fit(X)
    print(f'Preprocessing took {(time.time() - t0) * 1e3} ms.')
    for query_params in query_args:
        print(f'Running {algo} with {query_params}')
        results = list()
        est.set_query_param(query_params)
        # est.query(numpy.array(Y, dtype=numpy.float32))
        for rep in range(args.reps):
            results.append(est.query(Y))
        #try:
        processed_results = est.process_results(results)
        write_result(processed_results, args.dataset, 
            args.mu, args.query_set, algo, args.build_args, json.dumps(query_params),
            build_time=est.build_time)
        #except:
        #    print(f"Error processing {algo} with {query_params}")

if __name__ == "__main__":
    run_from_cmdline()