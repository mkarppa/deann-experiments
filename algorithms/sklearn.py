from algorithms.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors, KernelDensity
import numpy as np
import time
import pandas as pd

class SklearnKDTreeEstimator(BaseEstimator):
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.h = h
        self.est = KernelDensity(algorithm = 'kd_tree', bandwidth = h,
                                     kernel = kernel)

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
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.h = h
        self.est = KernelDensity(algorithm = 'ball_tree', bandwidth = h,
                                     kernel = kernel)

    def name(self):
        return 'sklearn-balltree'
