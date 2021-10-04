import os

from cmd_runner import run_from_cmdline
from install import build
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from itertools import product
from preprocess_datasets import get_dataset,DATASETS

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

def blacklist_algo(algo, build_args, query_args):
    f = open('blacklist.yaml', 'w')
    yaml.dump({algo : {"build" : build_args, "query": query_args}}, f)
    f.close()

def is_blacklisted(algo, build_args, query_args):
    if not os.path.exists("blacklist.yaml"):
        return False
    f = open('blacklist.yaml', 'rb')
    d = yaml.load(f.read())
    f.close()
    if algo not in d:
        return False
    if d[algo]["build"] != build_args:
        return False
    if d[algo]["query"] != query_args:
        return False

    return True

def run_docker(cpu_limit, mem_limit, dataset, algo, docker_tag, wrapper, constructor, reps, query_set, 
    build_args, query_args, bw, mu, timeout=3600, blacklist=False):
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

    client = docker.from_env()


    container = client.containers.run(
                docker_tag,
                cmd,
                volumes={
                    os.path.abspath('.'):
                        {'bind': '/home/app/', 'mode': 'rw'},
                },
                mem_limit=mem_limit,
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
        if exit_code not in [0, None]:
            print(container.logs().decode())
            print('Child process for container %s raised exception %d' % (container.short_id, exit_code))
            if blacklist:
                blacklist_algo(algo, build_args, query_args)
    except:
        print('Container.wait for container %s failed with exception' % container.short_id)
        print('Invoked with %s' % cmd)
        traceback.print_exc()
        if blacklist:
            blacklist_algo(algo, build_args, query_args)
    finally:
        container.remove(force=True)

def run_no_docker(cpu_limit, mem_limit, dataset, algo, docker_tag, wrapper, constructor, reps, query_set, 
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
    #os.system("python cmd_runner.py " + " ".join(cmd))

def run_worker(args, queue, i):
    while not queue.empty():
        algo, bw, algo_def, build_args, query_args = queue.get()
        mem_limit = int(8e9) # 8gb
        cpu_limit = i
        if args.blacklist and not args.force and \
            is_blacklisted(algo, build_args, query_args):
            print(f"BLACKLISTED: Not running {algo} with build={build_args} and query={query_args}")
            continue

        if not args.no_docker:
            run_docker(cpu_limit, mem_limit, args.dataset, algo, algo_def["docker"], algo_def["wrapper"], algo_def["constructor"], 
                args.reps, args.query_set, build_args, query_args, 
                bw, args.kde_value, args.timeout, args.blacklist)
        else:
            run_no_docker(cpu_limit, mem_limit, args.dataset, algo, algo_def["docker"], algo_def["wrapper"], algo_def["constructor"], 
                args.reps, args.query_set, build_args, query_args, 
                bw, args.kde_value)

def get_result_fn(dataset, mu, query_set, algo, args_str, query_str):
    dir_name = os.path.join("results", dataset, query_set, algo, str(mu))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    return os.path.join(dir_name, f'{args_str}_{query_str}.hdf5')

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


    tau = np.percentile(np.array(dataset[f'kde.{args.query_set}' + f'{mu:f}'.strip('0')], dtype=np.float32),
                            1)

    exps = {}

    # generate all experiments and remove the once that are already there
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
            queue.put((algo, bw, definitions[algo], params_dict["build"], json.dumps(params_dict["query"])))
    workers = [multiprocessing.Process(target=run_worker, args=(args, queue, i)) for i in range(args.cpu, args.cpu + 1)]
    [worker.start() for worker in workers]
    [worker.join() for worker in workers]

if __name__ == "__main__":
    main()



