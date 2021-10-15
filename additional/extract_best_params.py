#!/usr/bin/env python3

import pandas as pd
import sys
from math import ceil
import argparse

QUERY_SET = 'validation'
MAX_REL_ERR = 0.1

def extract_params(df, ds, mu, algo, k_equals_m, timeout):
    # only allow instances whose runtime is at most a factor of timeout away from naive
    timeout = ceil(df[(df['dataset'] == ds) & (df['algorithm'] == 'naive')]['query_time'].max())*timeout

    # print(ds,mu,algo)
    df = df[(df['dataset'] == ds) & \
            (df['mu'] == mu) & \
            (df['algorithm'] == algo) & \
            (df['rel_err'] < MAX_REL_ERR) & \
            (df['query_time'] < timeout)]

    if k_equals_m and algo in ['ann-faiss', 'ann-permuted-faiss']:
        df = df.copy()
        df['k'] = df['params'].str.split('_').map(lambda x: x[1].strip('[]')).str.split(', ').map(lambda x: x[0]).astype(int)
        df['m'] = df['params'].str.split('_').map(lambda x: x[1].strip('[]')).str.split(', ').map(lambda x: x[1]).astype(int)
        df = df[df['k'] == df['m']]

    df = df.sort_values(by='query_time')
    if df.shape[0] == 0:
        return None
    df = df.iloc[0]
    # print(df)
    # if df['algorithm'] in ['ann-permuted-faiss', 'ann-faiss']:
    #     params = list(map(lambda s: int(s.split('=')[1]),
    #                         df['params'].strip('()').split(', ')))
    #     params = dict(zip(['k','m','nlist','nprobe'], params))
    # elif df['algorithm'] in ['random-sampling', 'rsp']:
    #     params = { 'm' : int(df['params']) }
    # elif df['algorithm'] == 'naive':
    #     params = {}
    # elif df['algorithm'] in ['rs','hbe']:
    #     params = list(map(lambda s: float(s.split('=')[1]),
    #                           df['params'].strip('()').split(', ')))
    #     params = dict(zip(['eps', 'tau'], params))
    # elif df['algorithm'] in ['sklearn-balltree', 'sklearn-kdtree']:
    #     # print(df['params'])
    #     params = df['params'].strip('()').split(', ')
    #     params = (int(params[0].split('=')[1]), float(params[1].split('=')[1]),
    #                   float(params[2].split('=')[1]))
    #     params = dict(zip(['ls', 'atol', 'rtol'], params))
    # # print(df['params'])
    # # print(df['algorithm'])
    return df['params']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-equals-m',
                        action='store_true')
    parser.add_argument('--timeout',
                        type=int,
                        default=10)
    parser.add_argument('filename',
                        type=str,
                        metavar='results.csv') 
    args = parser.parse_args()

    filename = args.filename
    k_equals_m = args.k_equals_m
    timeout = args.timeout
    
    df = pd.read_csv(filename)
    df = df[df['query_set'] == QUERY_SET]
    datasets = df['dataset'].unique()
    mus = df['mu'].unique()    
    algorithms = df['algorithm'].unique()
    
    rows = list()
    for ds in datasets:
        for mu in mus:
            for algo in algorithms:
                row = {'dataset' : ds, 'mu' : mu, 'algorithm' : algo,
                       'params' : extract_params(df, ds, mu, algo, k_equals_m, timeout)}
                # for (k,v) in extract_params(df, ds, mu, algo).items():
                #     row[k] = v
                # print(row)
                print(row)                    
                rows.append(row)
    df = pd.DataFrame(rows)
    # print(df)
    df.to_csv('best_params.csv', index=None)
    
if __name__ == '__main__':
    main()
    
