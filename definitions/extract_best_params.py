#!/usr/bin/env python3

import pandas as pd

QUERY_SET = 'validation'
MAX_REL_ERR = 0.1

def extract_params(df, ds, mu, algo):   
    df = df[(df['dataset'] == ds) & \
                (df['mu'] == mu) & \
                (df['algorithm'] == algo) & \
                (df['rel_err'] < MAX_REL_ERR)] \
                .sort_values(by='query_time')
    # print(df)
    if df.shape[0] == 0:
        return {}
    df = df.iloc[0]
    if df['algorithm'] in ['ann-permuted-faiss', 'ann-faiss']:
        params = list(map(lambda s: int(s.split('=')[1]),
                            df['params'].strip('()').split(', ')))
        params = dict(zip(['k','m','nlist','nprobe'], params))
    elif df['algorithm'] in ['random-sampling', 'rsp']:
        params = { 'm' : int(df['params']) }
    elif df['algorithm'] == 'naive':
        params = {}
    elif df['algorithm'] in ['rs','hbe']:
        params = list(map(lambda s: float(s.split('=')[1]),
                              df['params'].strip('()').split(', ')))
        params = dict(zip(['eps', 'tau'], params))
    elif df['algorithm'] in ['sklearn-balltree', 'sklearn-kdtree']:
        # print(df['params'])
        params = df['params'].strip('()').split(', ')
        params = (int(params[0].split('=')[1]), float(params[1].split('=')[1]),
                      float(params[2].split('=')[1]))
        params = dict(zip(['ls', 'atol', 'rtol'], params))
    # print(df['params'])
    # print(df['algorithm'])
    return params

def main():
    df = pd.read_csv('amalgamated.csv')
    df = df[~df['params'].str.contains('atol=0.001,')]

    df = df[df['query_set'] == QUERY_SET]
    # print(df)
    datasets = df['dataset'].unique()
    # print(datasets)
    mus = df['mu'].unique()
    # print(mus)
    algorithms = df['algorithm'].unique()
    # print(algorithms)
    
    rows = list()
    for ds in datasets:
        for mu in mus:
            for algo in algorithms:
                # print(ds,mu,algo)
                row = {'dataset' : ds, 'mu' : mu, 'algorithm' : algo,
                           'k' : None, 'm' : None, 'nlist' : None,
                           'nprobe' : None, 'eps' : None, 'tau' : None,
                           'ls' : None, 'atol' : None, 'rtol' : None}
                for (k,v) in extract_params(df, ds, mu, algo).items():
                    row[k] = v
                rows.append(row)
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv('best_params.csv', index=None)
    
if __name__ == '__main__':
    main()
    
