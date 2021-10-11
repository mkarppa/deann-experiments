#!/usr/bin/env python3

import yaml
from math import log, sqrt, floor, ceil
from itertools import product
from operator import itemgetter
import numpy as np

if __name__ == '__main__':
    max_n = 3000000
    iv = 10
    multiplier = sqrt(2)
    max_i = int((log(max_n)-log(iv))/log(multiplier))
    mks = [int(round(iv*multiplier**i)) for i in range(max_i+1)]
    n_lists = [1<<i for i in range(5,13)]
    n_query = 1

    query_args = list(sorted([[mk,mk,n_list,1] for (mk,n_list) in product(mks,n_lists)], key=itemgetter(2)))

    epsilons = np.round(np.arange(1.5,0.05,-0.05),5).tolist()
    taus = np.round([0.01 / sqrt(2)**i for i in range(20)],5).tolist()
    query_args_hbe = list(sorted([[eps,tau] for (eps,tau) in product(epsilons,taus)], key=itemgetter(1), reverse=True))

    ls = [int(round(10*sqrt(2)**i)) for i in range(10)]
    trs = [round(0.05*i,4) for i in range(11)]
    query_args_sklearn = [[l,0.0,tr] for (l,tr) in product(ls,trs)]

    algos = {
        'naive' : {
            'constructor' : 'Naive',
            'wrapper'     : 'deann_wrapper',
            'docker'      : 'deann-experiments-deann' 
        },
        'ann-faiss' : {
            'constructor' : 'ANNFaiss',
            'query' : query_args,
            'wrapper': 'deann_wrapper',
            'docker'      : 'deann-experiments-deann'                         
        },
        'ann-permuted-faiss' : {
            'constructor' : 'ANNPermutedFaiss',
            'query' : query_args,
            'wrapper': 'deann_wrapper',
            'docker'      : 'deann-experiments-deann' 
        },
        'random-sampling' : {
            'constructor' : 'RandomSampling',
            'query' : mks,
            'wrapper': 'deann_wrapper',
            'docker'      : 'deann-experiments-deann'
        },
        'rsp' : {
            'constructor' : 'RandomSamplingPermuted',
            'query' : mks,
            'wrapper': 'deann_wrapper',
            'docker'      : 'deann-experiments-deann'
        },
        'hbe' : {
            'args' : { 'binary' : 'hbe' },
            'constructor' : 'HBEEstimator',
            'query' : query_args_hbe,
            'wrapper' : 'hbe',
            'docker' : 'deann-experiments-hbe',
            'separate-queries' : True
        },
        'rs' : {
            'args' : { 'binary' : 'hbe' },
            'constructor' : 'RSEstimator',
            'query' : query_args_hbe,
            'wrapper' : 'hbe',
            'docker' : 'deann-experiments-hbe',
            'separate-queries' : True
        },
        'sklearn-balltree' : {
            'constructor' : 'SklearnBallTreeEstimator',
            'query' : query_args_sklearn,
            'wrapper' : 'sklearn',
            'docker' : 'deann-experiments-sklearn'
        },
        'sklearn-kdtree' : {
            'constructor' : 'SklearnKDTreeEstimator',
            'query' : query_args_sklearn,
            'wrapper' : 'sklearn',
            'docker' : 'deann-experiments-sklearn'
        }
    }
    print(yaml.dump(algos))
    
