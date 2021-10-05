from algorithms.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import psutil
import numpy as np
import sklearn.preprocessing
import faiss

import deann as kde

import time
import pandas as pd 

# This is a modified copy of the following file:
# https://github.com/erikbern/ann-benchmarks/blob/master/ann_benchmarks/algorithms/base.py

class BaseANN(object):
    def done(self):
        pass

    def get_memory_usage(self):
        """Return the current memory usage of this algorithm instance
        (in kilobytes), or None if this information is not available."""
        # return in kB for backwards compatibility
        return psutil.Process().memory_info().rss / 1024

    def fit(self, X):
        pass

    def query(self, q, n):
        return []  # array of candidate indices

    def batch_query(self, X, n):
        self.res = []
        for q in X:
            self.res.append(self.query(q, n))

    def get_batch_results(self):
        return self.res

    def get_additional(self):
        return {}

    def __str__(self):
        return self.name


class BruteNN(BaseANN):
    def __init__(self, metric = 'euclidean', return_dists = True,
                     return_samples = True):
        if metric == 'euclidean':
            self._metric = 'euclidean'
        elif metric == 'taxicab':
            self._metric = 'manhattan'
        else:
            raise ValueError(f'invalid metric ``{metric}\'\' supplied')
        self._return_dists = return_dists
        self._return_samples = return_samples
    
    def fit(self, X):
        self._nn = NearestNeighbors(algorithm='brute', metric=self._metric)
        self._nn.fit(X)
        self._n = X.shape[0]

    def query(self, q, n):
        if self._return_dists and self._return_samples:
            return self._nn.kneighbors(q.reshape(1,-1),n) + (np.array([self._n], np.int32),)
        elif self._return_dists and not self._return_samples:
            return self._nn.kneighbors(q.reshape(1,-1),n)
        elif not self._return_dists and self._return_samples:
            return (self._nn.kneighbors(q.reshape(1,-1),n)[1], np.array([self._n], np.int32))
        else:
            return self._nn.kneighbors(q.reshape(1,-1),n)[1]



class Faiss(BaseANN):
    def query(self, v, n):
        s_before = self.get_additional()['dist_comps']
        if self._metric == 'angular':
            v /= np.linalg.norm(v)
        D, I = self.index.search(np.expand_dims(
            v, axis=0).astype(np.float32), n)
        s_after = self.get_additional()['dist_comps']
        return np.sqrt(D[0]), I[0], np.array([s_after-s_before], dtype=np.int32)

    def batch_query(self, X, n):
        if self._metric == 'angular':
            X /= np.linalg.norm(X)
        self.res = self.index.search(X.astype(np.float32), n)

    def get_batch_results(self):
        D, L = self.res
        res = []
        for i in range(len(D)):
            r = []
            for l, d in zip(L[i], D[i]):
                if l != -1:
                    r.append(l)
            res.append(r)
        return res

class FaissIVF(Faiss):
    def __init__(self, metric, n_list):
        self._n_list = n_list
        self._metric = metric

    def fit(self, X):
        if self._metric == 'angular':
            X = sklearn.preprocessing.normalize(X, axis=1, norm='l2')

        if X.dtype != np.float32:
            X = X.astype(np.float32)

        self.quantizer = faiss.IndexFlatL2(X.shape[1])
        index = faiss.IndexIVFFlat(
            self.quantizer, X.shape[1], self._n_list, faiss.METRIC_L2)
        index.train(X)
        index.add(X)
        self.index = index

    def set_query_arguments(self, n_probe):
        faiss.cvar.indexIVF_stats.reset()
        self._n_probe = n_probe
        self.index.nprobe = self._n_probe

    def get_additional(self):
        return {"dist_comps": faiss.cvar.indexIVF_stats.ndis +      # noqa
                faiss.cvar.indexIVF_stats.nq * self._n_list}

    def __str__(self):
        return 'FaissIVF(n_list=%d, n_probe=%d)' % (self._n_list,
                                                    self._n_probe)




class Naive(BaseEstimator):
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        print(f'constructing Naive with kernel={kernel}')
        self.est = kde.NaiveKde(h, kernel)

    def fit(self, X):
        t0 = time.time()
        print('fitting dataset')
        self.X = X
        self.est.fit(X)
        fit_time = time.time() - t0
        self.build_time = max(self.build_time, fit_time)

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
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.est = kde.RandomSampling(h, kernel, 1)
        self.rs = 1

    def set_query_param(self, param):
        self.rs = param
        self.est.reset_parameters(self.rs)

    def name(self):
        return 'random-sampling'

    def __str__(self):
        return f'{self.rs}'


class RandomSamplingPermuted(Naive):
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        self.est = kde.RandomSamplingPermuted(h, kernel, 1)
        self.rs = 1

    def set_query_param(self, param):
        self.rs = param
        self.est.reset_parameters(self.rs)

    def name(self):
        return 'rsp'

    def __str__(self):
        return f'{self.rs}'


class ANN(Naive):
    def __init__(self, dataset, query_set, kernel, h, args, ann_object):
        self.est = kde.AnnEstimator(h, kernel, 0, 0, ann_object)
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
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        super().__init__(dataset, query_set, kernel, h, args, FaissIVF('euclidean', None))
        self.X = None

    def set_query_param(self, param):
        t0 = time.time()
        super().set_query_param(param[:2])
        self.n_list = param[2]
        self.n_probe = param[3]
        if self.ann_object._n_list != self.n_list:
            self.ann_object._n_list = self.n_list
            if self.X is not None:
                self.ann_object.fit(self.X)
        query_param_time = time.time() - t0
        self.build_time = max(query_param_time, self.build_time)

    def fit(self, X):
        self.X = np.array(X, dtype=np.float32)
        t0 = time.time()
        self.est.fit(self.X)
        fit_time = time.time() - t0
        self.build_time = max(fit_time, self.build_time)


    def query(self, Y):
        self.ann_object.set_query_arguments(self.n_probe)
        return super().query(Y)

    def name(self):
        return 'ann-faiss'

    def __str__(self):
        return f'(nn={self.nn_k}, rs={self.rs_k}, n_list={self.n_list}, n_probe={self.n_probe})'



class ANNPermuted(Naive):
    def __init__(self, dataset, query_set, kernel, h, args, ann_object):
        self.est = kde.AnnEstimatorPermuted(h, kernel, 0, 0, ann_object)
        self.ann_object = ann_object

    def set_query_param(self, param):
        self.nn_k, self.rs_k = param
        self.est.reset_parameters(self.nn_k, self.rs_k)

    def fit(self, X):
        self.X = X
        self.est.fit(self.X)
        self.ann_object.fit(self.X)


class ANNPermutedFaiss(ANNPermuted):
    def __init__(self, dataset, query_set, kernel, mu, h, args):
        super().__init__(dataset, query_set, kernel, h, args, FaissIVF('euclidean', None))
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
