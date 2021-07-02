import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
from itertools import product
from preprocess_datasets import get_dataset,DATASETS

import sys
sys.path.append("deann")

from extern.faiss import FaissIVF
from extern.brute import BruteNN
import deann as kde

from sklearn.neighbors import NearestNeighbors, KernelDensity
import argparse
import yaml
import subprocess
import re
import h5py
import numpy as np
import time
import pandas as pd

from tempfile import NamedTemporaryFile
from datetime import datetime


class BaseEstimator:
    def fit(self, X):
        raise NotImplementedError()

    def query(self, Y):
        raise NotImplementedError()

    def set_query_param(self, args):
        raise NotImplementedError()

    def __str__(self):
        raise Exception("No representation given")

class RSEstimator(BaseEstimator):
    def __init__(self, dataset, query_set, mu, h, args):
        self.eps = None
        self.binary =  args['binary']
        self.dataset = dataset
        self.query_set = query_set
        self.mu = '{:f}'.format(mu).strip('0')
        self.h = h

        # if not os.path.exists(os.path.join("external", "rehashing", "resources", "data", f'{dataset}.{query_set}.conf')):
        #     print("Creating dataset")
        #     print(get_dataset(dataset))
        #     quit()
        #     from_hdf5.main(get_dataset(dataset), dataset)

    def fit(self, X):
        # pass # do nothing
        self.n = X.shape[0]
        self.d = X.shape[1]

    def query(self, Y):
        logfilename = f'{self.name()} {self.dataset} {self.mu} eps={self.eps} tau={self.tau} {datetime.now()}.log'
        self.m = Y.shape[0]
        with NamedTemporaryFile('w', delete = True) as f:
            self.write_conf(f)
            # self.result = subprocess.run(self.cmd().split(),
            #             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            # self.result = self.result.stdout.decode('utf-8').strip()
            # print('wrote config to', f.name)
            # print('cmd:',self.cmd(f.name))
            start = time.time()
            res = subprocess.run(self.cmd(f.name).split(),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE).stdout\
                                    .decode('utf-8').strip()
            end = time.time()
            print(f'{end-start}')
            print(f'writing log to {logfilename}')
            with open(logfilename, 'w') as f:
                f.write(res)
            return res


    def set_query_param(self, query_param):
        self.eps, self.tau = query_param


    def write_conf(self, f):
        data_dir = 'external/rehashing/resources/data'
        config  =  'exp {\n'
        config += f'    name = "{self.dataset}";\n'
        config += f'    fpath = "{data_dir}/{self.dataset}.train.csv";\n'
        config += f'    qpath = "{data_dir}/{self.dataset}.{self.query_set}'
        config += '.csv";\n'
        config += f'    exact_path = "{data_dir}/{self.dataset}.kde.'
        config += f'{self.query_set}{self.mu}.csv";\n'
        config += f'    kernel = "exponential";\n'
        config += f'    d = "{self.d}";\n'
        config += f'    n = "{self.n}";\n'
        config += f'    m = "{self.m}";\n'
        config += f'    h = "{self.h}";\n'
        config += f'    bw_const = "true";\n'
        config += f'    ignore_header = "false";\n'
        config += f'    start_col = "0";\n'
        config += f'    end_col = "{self.d-1}";\n'
        config += f'    eps = "{self.eps:f}";\n'
        config += f'    tau = "{self.tau:f}";\n'
        config +=  '    beta = "0.5";\n'
        config +=  '}\n'
        # with open(self.conf_filename,'w') as f:
        f.write(config)
        f.flush()

    def process_results(self, results):
        regex = r"RESULT id=(\d+) est=(\S+) samples=(.*) time=(\S+)"
        # ids = []
        # ests = []
        # samples = []
        # times = []
        # print(self.result.split("\n"))
        processed_results = {
            'iter' : [],
            'id' : [],
            'est' : [],
            'samples' : [],
            'time' : [],
        }

        # print(results)
        for i in range(len(results)):
            result = results[i]
            # print(result.split('\n'))
            for line in result.split('\n'): # self.result.split("\n"):
                # print(line)
                if m := re.match(regex, line):
                    # ids.append(int(m.group(1)))
                    # ests.append(float(m.group(2)))
                    # samples.append(int(m.group(3)))
                    # times.append(float(m.group(4)))
                    processed_results['iter'].append(i)
                    id = int(m.group(1))
                    processed_results['id'].append(id)
                    processed_results['est'].append(float(m.group(2)))
                    processed_results['samples'].append(float(m.group(3)))
                    processed_results['time'].append(float(m.group(4)))

        #return (ids, ests, samples, times)
        return pd.DataFrame(processed_results)

    def name(self):
        return 'rs'

    def cmd(self, conf_filename):
        return f'{self.binary} {conf_filename} exp {self.eps} true'

    def __str__(self):
        return f'(eps={self.eps}, tau={self.tau})'



class HBEEstimator(RSEstimator):
    def name(self):
        return 'hbe'

    def cmd(self, conf_filename):
        return f'{self.binary} {conf_filename} exp {self.eps}'



class SklearnKDTreeEstimator(BaseEstimator):
    def __init__(self, dataset, query_set, mu, h, args):
        self.h = h
        self.est = KernelDensity(algorithm = 'kd_tree', bandwidth = h,
                                     kernel = 'exponential')

    def set_query_param(self, query_param):
        self.leaf_size, self.atol, self.rtol = query_param
        self.est.set_params(leaf_size = self.leaf_size,
                            atol = self.atol,
                            rtol = self.rtol)

    def fit(self, X):
        self.n, d = X.shape
        temp_est = KernelDensity(bandwidth = self.h, kernel = 'exponential')
        temp_est.fit(np.zeros((1,d)))
        self.constant = -temp_est.score(np.zeros((1,d)))
        self.est.fit(X)

    def query(self, Y):
        t0 = time.time()
        Z = np.exp(self.est.score_samples(Y) + self.constant)
        t1 = time.time()
        print(t1-t0)
        return {
            'm' : Y.shape[0],
            'est' : Z,
            'samples' : [self.n]*Y.shape[0],
            'total-time' : (t1 - t0) * 1000,
        }

    def name(self):
        return 'sklearn-kdtree'

    def process_results(self, results):
        processed_results = {
            'iter' : [],
            'id' : [],
            'est' : [],
            'samples' : [],
            'time' : [],
        }
        m = results[0]['m']
        for i in range(len(results)):
            assert results[i]['m'] == m

        for i in range(len(results)):
            result = results[i]
            for j in range(m):
                processed_results['iter'].append(i)
                processed_results['id'].append(j)
                processed_results['est'].append(result['est'][j])
                processed_results['samples'].append(result['samples'][j])
                processed_results['time'].append(result['total-time']/m)

        return pd.DataFrame(processed_results)

    def __str__(self):
        return f'(leaf_size={self.leaf_size}, atol={self.atol}, rtol={self.rtol})'


class SklearnBallTreeEstimator(SklearnKDTreeEstimator):
    def __init__(self, dataset, query_set, mu, h, args):
        self.h = h
        self.est = KernelDensity(algorithm = 'ball_tree', bandwidth = h,
                                     kernel = 'exponential')

    def name(self):
        return 'sklearn-balltree'


class Naive(BaseEstimator):
    def __init__(self, dataset, query_set, mu, h, args):
        self.est = kde.NaiveKde(h, 'exponential')

    def fit(self, X):
        print('fitting dataset')
        self.X = X
        self.est.fit(X)

    def query(self, Y):
        t0 = time.time()
        (Z, S) = self.est.query(Y)
        t1 = time.time()
        print(t1-t0)

        #self.res =
        return {
            'm' : Y.shape[0],
            'est' : Z,
            'samples' : S,
            'total-time' : (t1 - t0) * 1000,
        }
        # print(self.res['est'])

    def process_results(self, results):
        m = results[0]['m']
        for res in results:
            assert res['m'] == m

        processed_results = {
            'iter' : [],
            'id' : [],
            'est' : [],
            'samples' : [],
            'time' : [],
        }
        for i in range(len(results)):
            for j in range(m):
                processed_results['iter'].append(i)
                processed_results['id'].append(j)
                processed_results['est'].append(results[i]['est'][j])
                processed_results['samples'].append(results[i]['samples'][j])
                processed_results['time'].append(results[i]['total-time']/m)

        # ids = numpy.array(range(m))
        #ests = self.res['est']
        # samples = numpy.array([X.shape[0] for _ in range(m)])
        #samples = self.res['samples']
        #times = numpy.array([self.res['total-time']] * m) / m

        # return (ids, ests, samples, times)

        return pd.DataFrame(processed_results)

    def name(self):
        return 'naive'

    def __str__(self):
        return 'default'

    def set_query_param(self, args):
        self.est.reset_parameters(args)



class RandomSampling(Naive):
    def __init__(self, dataset, query_set, mu, h, args):
        self.est = kde.RandomSampling(h, 'exponential', 1)
        self.rs = 1

    def set_query_param(self, param):
        self.rs = param
        self.est.reset_parameters(self.rs)

    def name(self):
        return 'random-sampling'

    def __str__(self):
        return f'{self.rs}'



class RandomSamplingPermuted(Naive):
    def __init__(self, dataset, query_set, mu, h, args):
        self.est = kde.RandomSamplingPermuted(h, 'exponential', 1)
        self.rs = 1

    def set_query_param(self, param):
        self.rs = param
        self.est.reset_parameters(self.rs)

    def name(self):
        return 'rsp'

    def __str__(self):
        return f'{self.rs}'



class ANN(Naive):
    def __init__(self, dataset, query_set, h, args, ann_object):
        self.est = kde.AnnEstimator(h, "exponential", 0, 0, ann_object)
        self.ann_object = ann_object

    def set_query_param(self, param):
        self.nn_k, self.rs_k = param
        self.est.reset_parameters(self.nn_k, self.rs_k)
        # nn = BruteNN()
        # self.est = kde.AnnEstimator(self.h, "exponential", self.nn_k, self.rs_k, nn)
        # super ugly
        # if self.fitted:
        #     nn.fit(self.X)
        #     self.est.fit(self.X)

    def fit(self, X):
        # self.X = numpy.array(X, dtype=numpy.float32)
        # self.fitted = True
        self.X = X
        self.est.fit(self.X)
        self.ann_object.fit(self.X)


class ANNBrute(ANN):
    def __init__(self, dataset, query_set, mu, h, args):
        # self.est = kde.AnnEstimator(h, "exponential", 0, 0, BruteNN())
        # self.h = h
        # self.fitted = False
        super().__init__(dataset, query_set, h, args, BruteNN())


    def name(self):
        return 'ann-brute'

    def __str__(self):
        return f'(nn={self.nn_k}, rs={self.rs_k})'



class ANNFaiss(ANN):
    def __init__(self, dataset, query_set, mu, h, args):
        super().__init__(dataset, query_set, h, args, FaissIVF('euclidean', None))
        self.X = None

    def set_query_param(self, param):
        super().set_query_param(param[:2])
        self.n_list = param[2]
        self.n_probe = param[3]
        if self.ann_object._n_list != self.n_list:
            self.ann_object._n_list = self.n_list
            if self.X is not None:
                self.ann_object.fit(self.X)

    def fit(self, X):
        self.X = np.array(X, dtype=np.float32)
        self.est.fit(self.X)


    def query(self, Y):
        self.ann_object.set_query_arguments(self.n_probe)
        return super().query(Y)

    def name(self):
        return 'ann-faiss'

    def __str__(self):
        return f'(nn={self.nn_k}, rs={self.rs_k}, n_list={self.n_list}, n_probe={self.n_probe})'



class ANNPermuted(Naive):
    def __init__(self, dataset, query_set, h, args, ann_object):
        self.est = kde.AnnEstimatorPermuted(h, "exponential", 0, 0, ann_object)
        self.ann_object = ann_object

    def set_query_param(self, param):
        self.nn_k, self.rs_k = param
        self.est.reset_parameters(self.nn_k, self.rs_k)

    def fit(self, X):
        self.X = X
        self.est.fit(self.X)
        self.ann_object.fit(self.X)


class ANNPermutedFaiss(ANNPermuted):
    def __init__(self, dataset, query_set, mu, h, args):
        super().__init__(dataset, query_set, h, args, FaissIVF('euclidean', None))
        self.X = None

    def set_query_param(self, param):
        super().set_query_param(param[:2])
        self.n_list = param[2]
        self.n_probe = param[3]
        if self.ann_object._n_list != self.n_list:
            self.ann_object._n_list = self.n_list
            if self.X is not None:
                self.ann_object.fit(self.X)

    def fit(self, X):
        self.X = np.array(X, dtype=np.float32)
        self.est.fit(self.X)


    def query(self, Y):
        self.ann_object.set_query_arguments(self.n_probe)
        return super().query(Y)

    def name(self):
        return 'ann-permuted-faiss'

    def __str__(self):
        return f'(nn={self.nn_k}, rs={self.rs_k}, n_list={self.n_list}, n_probe={self.n_probe})'



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

    print(f"Running on {args.dataset}")
    dataset_name = args.dataset
    dataset = get_dataset(dataset_name)
    print(dataset)

    mu = args.kde_value
    kde_str = 'kde.' + args.query_set + '{:f}'.format(mu).strip('0')
    _, bw = dataset.attrs[kde_str]

    print(f"Running with bandwidth {bw} to achieve kde value {mu}.")


    with open(args.definition, 'r') as f:
        definitions = yaml.load(f, Loader=yaml.Loader)

    print(definitions)

    if args.list_algorithms:
        print("Available algorithms:")
        print("\n".join(definitions))
        exit(0)

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
        Est_class = getattr(sys.modules[__name__], definitions[algo]['constructor'])
        est = Est_class(dataset_name, args.query_set, mu, bw, definitions[algo].get('args', {}))
        for query_param in definitions[algo].get('query', [None]):
            est.set_query_param(query_param)
            if args.force or not os.path.exists(get_result_fn(dataset_name, mu, args.query_set, est)):
                exps.setdefault(algo, []).append(query_param)



    print(exps)

# generate all experiments and remove the once that are already there
    for algo, query_params in exps.items():
        Est_class = getattr(sys.modules[__name__], definitions[algo]['constructor'])
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
            processed_results = est.process_results(results)
            # write_result(dataset_name, mu, est, args.query_set, Y.shape[0])
            write_result(processed_results, dataset_name, mu, est, args.query_set, Y.shape[0])



if __name__ == "__main__":
    main()



