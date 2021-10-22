#!/usr/bin/env python3

import pandas as pd
import sys
import yaml

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.stderr.write(f'usage: {sys.argv[0]} <best_params.csv>\n')
        quit(1)
    
    df = pd.read_csv(sys.argv[1])
    datasets = df['dataset'].unique()
    mus = df['mu'].unique()
    algorithms = df['algorithm'].unique()

    for ds in datasets:
        for mu in mus:
            algos = dict()
            for algo in algorithms:
                row = df[(df['dataset'] == ds) & (df['mu'] == mu) & (df['algorithm'] == algo)]
                assert row.shape[0] == 1
                row = row.iloc[0]
                if pd.isna(row['params']):
                    continue
                if algo in ['ann-permuted-faiss', 'ann-faiss']:
                    query_args = [list(map(int,row['params'].split('_')[1].strip('[]').split(',')))]
                    constructor = 'ANNFaiss' if algo == 'ann-faiss' else 'ANNPermutedFaiss'
                    algos[algo] = {
                        'constructor' : constructor,
                        'query' : query_args,
                        'wrapper': 'deann_wrapper',
                        'docker'      : 'deann-experiments-deann'                         
                    }
                elif algo == 'naive':
                    algos[algo] = {
                        'constructor' : 'Naive',
                        'wrapper'     : 'deann_wrapper',
                        'docker'      : 'deann-experiments-deann' 
                    }
                elif algo in ['random-sampling', 'rsp']:
                    query_args = [int(row['params'].split('_')[1])]
                    constructor = 'RandomSampling' if algo == 'random-sampling' else 'RandomSamplingPermuted'
                    algos[algo] = {
                        'constructor' : constructor,
                        'query' : query_args,
                        'wrapper': 'deann_wrapper',
                        'docker'      : 'deann-experiments-deann'
                    }
                elif algo == 'askit':
                    query_args = row['params'].split('_')[1].strip('[]').split(',')
                    assert len(query_args) == 8
                    query_args = [[int(query_args[0]), float(query_args[1]), int(query_args[2]),
                                   int(query_args[3]), int(query_args[4]), int(query_args[5]),
                                   int(query_args[6]), int(query_args[7])]]
                    assert len(query_args) == 1
                    assert len(query_args[0]) == 8
                    algos[algo] = {
                        'constructor' : 'Askit',
                        'docker' : 'deann-experiments-askit',
                        'query' : query_args,
                        'separate-queries': True,
                        'wrapper' : 'askit'
                    }
                elif algo in ['sklearn-balltree', 'sklearn-kdtree']:
                    constructor = 'SklearnBallTreeEstimator' if algo == 'sklearn-balltree' else 'SklearnKDTreeEstimator'
                    query_args = row['params'].split('_')[1].strip('[]').split(',')
                    query_args = [[int(query_args[0]), float(query_args[1]), float(query_args[2])]]
                    algos[algo] = {
                        'constructor' : constructor,
                        'query' : query_args,
                        'wrapper' : 'sklearn',
                        'docker' : 'deann-experiments-sklearn'
                    }
                elif algo in ['hbe', 'rs']:
                    constructor = 'HBEEstimator' if algo == 'hbe' else 'RSEstimator'
                    query_args = [list(map(float,row['params'].split('_')[1].strip('[]').split(',')))]
                    algos[algo] = {
                        'args' : { 'binary' : 'hbe' },
                        'constructor' : constructor,
                        'wrapper' : 'hbe',
                        'docker' : 'deann-experiments-hbe',
                        'query' : query_args,
                        'separate-queries' : True
                    }
                else:
                    assert False

            mustr = f'{mu:f}'.rstrip('0')
            with open(f'definitions/best_params_{ds}_{mustr}.yaml','w') as f:
                f.write(yaml.dump(algos))
