#!/usr/bin/env python3

import numpy as np
import pandas as pd
import argparse


def show_best(df, by):
    bests = list()
    algos = set(df['algorithm'])
    for algo in algos:
        bests.append(df[df['algorithm'] == algo].sort_values(by=by).iloc[0])
    df = pd.DataFrame(bests).sort_values(by='query_time')
    print(df.to_string())


    
def show_all(df,by):
    print(df.sort_values(by=by).to_string())

    
    
def main():   
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mu',
        default=None,
        type=float
    )
    parser.add_argument(
        '--algorithm',
        default=None,
        type=str
    )
    parser.add_argument(
        '--dataset',
        default=None,
        type=str
    )
    parser.add_argument(
        'action',
        type=str,
        choices=['show-best','show-all']
    )
    parser.add_argument(
        '--max-err',
        default=None,
        type=float
    )
    parser.add_argument(
        '-i',
        default = 'results.csv',
        type=str
    )
    parser.add_argument(
        '--query-set',
        default = None,
        type=str,
        choices=['test','validation']
    )
    parser.add_argument(
        '--sort-by',
        default = 'query_time',
        type=str
    )
    args = parser.parse_args()

    df = pd.read_csv(args.i)

    if args.mu is not None:
        df = df[df['mu'] == args.mu]
    if args.dataset is not None:
        df = df[df['dataset'] == args.dataset]
    if args.max_err is not None:
        df = df[df['rel_err'] < args.max_err]
    if args.algorithm is not None:
        df = df[df['algorithm'] == args.algorithm]
    if args.query_set is not None:
        df = df[df['query_set']] == args.query_set
    
    if args.action == 'show-best':
        show_best(df, args.sort_by)
    if args.action == 'show-all':
        show_all(df, args.sort_by)

if __name__ == '__main__':
    main()
