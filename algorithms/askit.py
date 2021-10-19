from datetime import datetime
from tempfile import NamedTemporaryFile

from datetime import datetime
import os
import pandas as pd
import re
import subprocess
import time
import math

from algorithms.base import BaseEstimator

class Askit(BaseEstimator):

    # TODO: Askit requires groundtruth k-nn files.
    # They are currently computed through a different workflow.
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.data_dir = os.path.join(os.getcwd(), "data")
        self.dataset = dataset
        self.query_set = query_set
        self.mu = mu
        self.h = h

    def fit(self, X):
        self.n = X.shape[0]
        self.d = X.shape[1]

    def query(self, Y):
        logfilename = f'{self.name()} {self.dataset} {self.mu} {datetime.now()}.log'
        with NamedTemporaryFile('w', delete=True) as f:
            start = time.time()
            print(" ".join(self.cmd()))
            res = subprocess.run(self.cmd(),
                stdout=subprocess.PIPE, check=True)
            end = time.time()

            res = res.stdout.decode('utf-8').strip()

            print(f'{end -start}s')

            with open(logfilename, 'w') as f:
                f.write(res)
            return res

    def set_query_param(self, args):
        self.k, self.id_tol, self.max_points, self.oversampling, self.id_rank, self.num_skel_targets, self.num_uniform_required, self.min_skel_level = args

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
                m = re.match(regex, line)
                if m:
                    processed_results['iter'].append(i)
                    id = int(m.group(1))
                    processed_results['id'].append(id)
                    processed_results['est'].append(float(m.group(2)))
                    processed_results['samples'].append(float(m.group(3)))
                    processed_results['time'].append(1000 * float(m.group(4))) # askit reports in seconds
                m = re.match(r"BUILD TIME: (\S+)", line)
                if m:
                    self.build_time = float(m.group(1))
        return pd.DataFrame(processed_results)

    def name(self):
        return 'askit'

    def cmd(self,):
        cmd = ['askit_deann_wrapper.exe',
            '-training_data', f'{self.data_dir}/{self.dataset}.train.data',
            '-test_data', f'{self.data_dir}/{self.dataset}.{"test" if self.query_set == "test" else "validation"}.data',
            '-training_knn_file', f'{self.data_dir}/{self.dataset}.train.knn',
            '-test_knn_file', f'{self.data_dir}/{self.dataset}.{"test" if self.query_set == "test" else "validation"}.knn',
            '-d', f'{self.d}',
            '-training_N', f'{self.n}',
            '-h', f'{self.h}',
            '-max_points', f'{self.max_points}',
            '-oversampling', f'{self.oversampling}',
            '-id_tol', f'{self.id_tol}',
            '-id_rank', f'{self.id_rank}',
            '-num_skel', f'{self.num_skel_targets}',
            '-num_uniform', f'{self.num_uniform_required}',
            '-min_skel_level', f'{self.min_skel_level}',
            '-test_N', f'500']

        return cmd

    def __str__(self):
        return 'askit'
