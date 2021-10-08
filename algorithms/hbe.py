from algorithms.base import BaseEstimator
from datetime import datetime
import time
import subprocess
import pandas as pd
import os
import re
from tempfile import NamedTemporaryFile
from preprocess_datasets import get_dataset_fn
import from_hdf5
from math import sqrt

class RSEstimator(BaseEstimator):
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.eps = None
        self.binary =  args['binary']
        self.dataset = dataset
        self.query_set = query_set
        self.mu = '{:f}'.format(mu).strip('0')
        self.h = h
        self.kernel = kernel
        from_hdf5.create_dataset(get_dataset_fn(dataset), kernel)

    def fit(self, X):
        # pass # do nothing
        self.n = X.shape[0]
        self.d = X.shape[1]

    def query(self, Y):
        logfilename = f'{self.name()} {self.dataset} {self.mu} kernel={self.kernel} eps={self.eps} tau={self.tau} {datetime.now()}.log'
        self.m = Y.shape[0]
        with NamedTemporaryFile('w', delete = True) as f:
            self.write_conf(f)
            # self.result = subprocess.run(self.cmd().split(),
            #             stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            # self.result = self.result.stdout.decode('utf-8').strip()
            # print('wrote config to', f.name)
            # print('cmd:',self.cmd(f.name))
            start = time.time()
            print(self.cmd(f.name).split())
            res = subprocess.run(self.cmd(f.name).split(),
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    check=True)

            res = res.stdout.decode('utf-8').strip()
            end = time.time()
            print(f'{end-start}')
            print(f'writing log to {logfilename}')
            with open(logfilename, 'w') as f:
                f.write(res)
            return res


    def set_query_param(self, query_param):
        self.eps, self.tau = query_param


    def write_conf(self, f):
        data_dir = os.path.join(os.environ['HBE'], 'resources', 'data')
        config  = ('exp' if self.kernel == 'exponential' else self.kernel) + ' {\n'
        config += f'    name = "{self.dataset}";\n'
        config += f'    fpath = "{data_dir}/{self.dataset}.train.csv";\n'
        config += f'    qpath = "{data_dir}/{self.dataset}.{self.query_set}'
        config += '.csv";\n'
        config += f'    exact_path = "{data_dir}/{self.dataset}.kde.'
        config += f'{self.query_set}{self.mu}.csv";\n'
        # config += f'    kernel = "exponential";\n'
        config += f'    kernel = "{self.kernel}";\n'
        config += f'    d = "{self.d}";\n'
        config += f'    n = "{self.n}";\n'
        config += f'    m = "{self.m}";\n'
        config += f'    h = "{self.h*sqrt(2) if self.kernel == "gaussian" else self.h}";\n'
        config += f'    bw_const = "true";\n'
        config += f'    ignore_header = "false";\n'
        config += f'    start_col = "0";\n'
        config += f'    end_col = "{self.d-1}";\n'
        config += f'    eps = "{self.eps:f}";\n'
        config += f'    tau = "{self.tau:f}";\n'
        config +=  '    beta = "0.5";\n'
        config +=  '}\n'
        f.write(config)
        f.flush()

    def process_results(self, results):
        regex = r"RESULT id=(\d+) est=(\S+) samples=(.*) time=(\S+)"
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
                    processed_results['iter'].append(i)
                    id = int(m.group(1))
                    processed_results['id'].append(id)
                    processed_results['est'].append(float(m.group(2)))
                    processed_results['samples'].append(float(m.group(3)))
                    processed_results['time'].append(float(m.group(4)))
                if m := re.match(r"Adaptive Table Init: (\S+)", line):
                    self.build_time = float(m.group(1))
        return pd.DataFrame(processed_results)

    def name(self):
        return 'rs'

    def cmd(self, conf_filename):
        scope = 'exp' if self.kernel == 'exponential' else self.kernel
        return f'{self.binary} {conf_filename} {scope} {self.eps} true'

    def __str__(self):
        return f'(eps={self.eps}, tau={self.tau})'



class HBEEstimator(RSEstimator):
    def name(self):
        return 'hbe'

    def cmd(self, conf_filename):
        scope = 'exp' if self.kernel == 'exponential' else self.kernel
        return f'{self.binary} {conf_filename} {scope} {self.eps}'


