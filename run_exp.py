import os

from cmd_runner import run_from_cmdline
from install import build
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from itertools import product
from preprocess_datasets import get_dataset,DATASETS
from result import get_result_fn, result_exists, write_result

import argparse
import yaml
import h5py
import multiprocessing
import threading
import numpy as np
import pandas as pd
import time
import json
import docker
import traceback
import psutil

from hacks import filter_run, filter_runs

def blacklist_algo(algo, build_args, query_args, args, err):
    # blacklist the first query argument which doesn't exist as a file
    # the assumption is that this is the one that produced an error and should not be re-run
    if len(query_args) == 0:
        return []
    while True:
        qa = json.dumps(query_args.pop(0))
        if result_exists(args.dataset, args.kde_value, args.query_set, algo, build_args, qa):
            continue
        write_result(None, args.dataset, args.kde_value, args.query_set,
            algo, build_args, qa,err=err)
        break
    # return the set of query parameters not inspected so far to continue running
    # experiments
    return query_args

def run_docker(cpu_limit, mem_limit, dataset, algo, kernel, docker_tag, wrapper, constructor, reps, query_set,
    build_args, query_args, bw, mu, timeout=3600, blacklist=False, separate_queries=False, args=None):

    query_args = json.loads(query_args)

    print('separate_queries:', separate_queries)
    print('timeout',timeout)

    while len(query_args) > 0:
        if separate_queries:
            run = query_args.pop(0)
            query_args_str = json.dumps([run])
            if filter_run(algo, dataset, query_set, mu, run):
                continue
        else:
            query_args = filter_runs(algo, dataset, query_set, mu, query_args)
            query_args_str = json.dumps(query_args)

        # quit()
        
        cmd = ['--dataset', dataset,
            '--algorithm', algo,
            '--wrapper', wrapper,
            '--kernel', kernel,
            '--constructor', constructor,
            '--mu', str(mu),
            '--bw', str(bw),
            '--reps', str(reps),
            '--query-set', query_set,
            '--build-args', '' + build_args + '',
            '--query-args', '' + query_args_str + '']

        client = docker.from_env()

        print(cmd)

        container = client.containers.run(
                    docker_tag,
                    cmd,
                    volumes={
                        os.path.abspath('.'):
                            {'bind': '/home/app/', 'mode': 'rw'},
                    },
                    mem_limit=mem_limit,
                    memswap_limit=mem_limit,
                    mem_swappiness=0,
                    cpuset_cpus=str(cpu_limit),
                    detach=True)

        print('Created container %s: CPU limit %s, mem limit %s, timeout %d, command %s' % \
                    (container.short_id, cpu_limit, mem_limit, timeout, cmd))

        def stream_logs():
            for line in container.logs(stream=True):
                print(line.decode().rstrip())

        t = threading.Thread(target=stream_logs, daemon=True)
        t.start()

        try:
            exit_code = container.wait(timeout=timeout)
            if type(exit_code) == dict:
                exit_code = exit_code["StatusCode"]

            # Exit if exit code
            assert exit_code in [0, None]
        except:
            print('Container.wait for container %s failed with exception' % container.short_id)
            print('Invoked with %s' % cmd)
            traceback.print_exc()
            if blacklist:
                if separate_queries:
                    blacklist_algo(algo, build_args, json.loads(query_args_str), args, err=container.logs().decode())
                else:
                    query_args = blacklist_algo(algo, build_args, query_args, args, err=container.logs().decode())
                continue
        finally:
            container.remove(force=True)
        if not separate_queries or len(query_args) == 0:
            break

def run_no_docker(cpu_limit, mem_limit, dataset, algo, kernel, docker_tag, wrapper, constructor, reps, query_set,
    build_args, query_args, bw, mu):
    cmd = ['--dataset', dataset,
           '--algorithm', algo,
           '--wrapper', wrapper,
           '--constructor', constructor,
           '--mu', str(mu),
           '--bw', str(bw),
           '--reps', str(reps),
           '--query-set', query_set,
           '--build-args', '' + build_args + '',
           '--query-args', '' + query_args + '']

    print(" ".join(cmd))
    run_from_cmdline(cmd)

def run_worker(args, queue, i):
    while not queue.empty():
        algo, bw, kernel, algo_def, build_args, query_args = queue.get()
        avail_mem = psutil.virtual_memory().available
        mem_limit = min(avail_mem, int(32e9)) # use max 32gb

        cpu_limit = i

        if not args.no_docker:
            run_docker(cpu_limit, mem_limit, args.dataset, algo, kernel, algo_def["docker"], algo_def["wrapper"], algo_def["constructor"],
                args.reps, args.query_set, build_args, query_args,
                bw, args.kde_value, args.timeout, args.blacklist, algo_def.get("separate-queries", False), args)
        else:
            run_no_docker(cpu_limit, mem_limit, args.dataset, algo, kernel, algo_def["docker"], algo_def["wrapper"], algo_def["constructor"],
                args.reps, args.query_set, build_args, query_args,
                bw, args.kde_value)

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
    parser.add_argument(
        '--no-docker',
        action='store_true'
    )
    parser.add_argument(
        '--timeout',
        help='timeout in seconds',
        default=3600,
        type=int,
    )
    parser.add_argument(
        '--blacklist',
        action="store_true"
    )
    parser.add_argument(
        '--cpu',
        type=int,
        default=0
    )
    parser.add_argument(
        '--kernel',
        choices=['exponential','gaussian'],
        default='gaussian'
    )
    args = parser.parse_args()

    with open(args.definition, 'r') as f:
        definitions = yaml.load(f, Loader=yaml.Loader)

    print(definitions)

    kernel = args.kernel

    if args.list_algorithms:
        print("Available algorithms:")
        print("\n".join(definitions))
        exit(0)

    print(f"Running on {args.dataset}")
    dataset_name = args.dataset
    dataset = get_dataset(dataset_name, kernel)
    print(dataset)

    mu = args.kde_value
    kde_str = 'kde.' + args.query_set + f'.{kernel}' + '{:f}'.format(mu).strip('0')
    _, bw = dataset.attrs[kde_str]

    print(f"Running with bandwidth {bw} to achieve kde value {mu}.")


    if args.algorithm:
        algorithms = [args.algorithm]
    else:
        algorithms = list(definitions.keys())


    #tau = np.percentile(np.array(dataset[f'kde.{args.query_set}' + f'{mu:f}'.strip('0')], dtype=np.float32),
     #                       1)

    exps = {}

    # generate all experiments and remove the ones that are already there
    for algo in algorithms:
        _args = definitions[algo].get('args', {})
        args_str = json.dumps(_args)
        exps[algo] = {
            "build": args_str,
            "query": [],
        }
        for query_params in definitions[algo].get('query', [None]):
            if type(query_params) == list and type(query_params[0]) == list:
                qps = product(*query_params)
            else:
                qps = [query_params]
            for qp in qps:
                query_str = json.dumps(qp)
                if args.force or not os.path.exists(get_result_fn(dataset_name, mu, args.query_set, algo, args_str, query_str)):
                    exps[algo]["query"].append(qp)
        if len(exps[algo]["query"]) == 0:
            del exps[algo]
    print(exps)

    if len(exps) == 0:
        print("No experiments to run.")
        exit(-1)

    queue = multiprocessing.Queue()
    for algo, params_dict in exps.items():
            queue.put((algo, bw, kernel, definitions[algo], params_dict["build"], json.dumps(params_dict["query"])))
    workers = [multiprocessing.Process(target=run_worker, args=(args, queue, i)) for i in range(args.cpu, args.cpu + 1)]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

if __name__ == "__main__":
    main()



