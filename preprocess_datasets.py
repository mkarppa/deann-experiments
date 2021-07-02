# This code is adapted from github.com/erikbern/ann-benchmarks.

import h5py
import numpy as np
import os
import random
import sys
import time
import argparse
from sklearn.model_selection import train_test_split
try:
        from urllib import urlretrieve
        from urllib import urlopen
except ImportError:
        from urllib.request import urlretrieve
        from urllib.request import urlopen

def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst)


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s.hdf5' % dataset)

def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://itu.dk/people/maau/kde/datasets/TODO%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            create_dataset(which)


    hdf5_f = h5py.File(hdf5_fn, 'r')
    return hdf5_f


# Everything below this line is related to creating datasets
def compute_norm(X, lens, q):
    v = np.maximum(np.min(lens + (q**2).sum(-1) - 2 * np.dot(X, q)), 0.0)
    return np.sqrt(v)

def compute_kde(X, lens, q, b):
    #print(min(lens + (q**2).sum(-1) - 2 * np.dot(X, q)))
    v = np.maximum(lens + (q**2).sum(-1) - 2 * np.dot(X, q),0.0)
    # print(v)
    return np.mean(np.exp(-np.sqrt(v)/b))

def batch_kde(X, X_sq_norms, Q, Q_sq_norms, b):
    return np.mean(np.exp(-np.sqrt(np.maximum(X_sq_norms[None,:] + Q_sq_norms[:,None] - 2*Q.dot(X.T), 0.0))/b), -1)

def scan_for_kde(X, Y, lower, upper, target=0.2):
    kde_vals = np.zeros(len(Y))
    lens = (X**2).sum(-1)
    left, right = lower, upper
    while True:
        dist = (left + right) / 2
        print(f'{lower} <= {left} <= {dist} <= {right} <= {upper}')
        print(f"Testing {dist}")
        for i in range(len(Y)):
            kde_vals[i] = compute_kde(X, lens, Y[i], dist)
        kde_val = np.median(kde_vals)
        print(f"got median kde value of {kde_val}")
        print(f'|{target}-{kde_val}|/{target} = {abs(target - kde_val)/target}')
        if abs(target - kde_val)/target <= 0.01:
            break
        elif target > kde_val:
            left = dist
        else:
            right = dist

    return kde_val, kde_vals, dist

def write_output(train, validation, test, fn, compute_bandwidth=False): #, compute_nn=False,
    #                     compute_dists=False):
    # normalization following Standford et al.
    #    from scipy import stats
    #    Y = stats.zscore(X, axis=0)
    # from sklearn.model_selection import train_test_split
    # queries=500
    # data, query = train_test_split(X, test_size=queries, random_state=42)

    assert train.dtype == np.float64
    assert validation.dtype == np.float64
    assert test.dtype == np.float64
    queries = test.shape[0]
    assert queries == 500
    assert validation.shape[0] == queries

    f = h5py.File(fn, 'w')
    #f.create_dataset('data', (len(data), len(data[0])),
	#dtype=data.dtype)[:] = data
    f.create_dataset('train', data = train)
    # f.create_dataset('query', (len(query), len(query[0])),
	# dtype=query.dtype)[:] = query
    f.create_dataset('validation', data = validation)
    f.create_dataset('test', data = test)

    # if compute_nn:
    #     from sklearn.neighbors import NearestNeighbors
    #     start = time.time()
    #     nn = NearestNeighbors(algorithm='brute')
    #     nn.fit(data)
    #     end = time.time()
    #     print(f'nn construction took {end-start} s')

    #     start = time.time()
    #     q = query[:2,:]
    #     (nn_dist, nn_ind) = nn.kneighbors(query,1000)
    #     end = time.time()
    #     print(f'nn query took {end-start} s')

    #     f.create_dataset('query.nn', data=nn_ind)

    #     # free memory
    #     del nn_ind
    #     del nn_dist
    #     del nn


    # compute norms

    if compute_bandwidth:
        s = time.time()
        train_lengths = (train**2).sum(-1)
        test_lengths = (test**2).sum(-1)
        validation_lengths = (validation**2).sum(-1)

        nn_dist = np.zeros(queries)
        for i in range(queries):
            nn_dist[i] = compute_norm(train, train_lengths, validation[i])

        # choose starting bandwidth based on median NN distance
        med_dist = np.median(nn_dist)

        del nn_dist
        # del query_lengths
        # del data_lengths

        print(f"Compute bandwidth based on median distance of {med_dist}")

        kde, kde_vals, b = scan_for_kde(train, validation, lower=med_dist/10,
            upper=10*med_dist,target=0.01)
        print(f"{kde} with b={b} for target={0.01}")
        f.attrs['kde.validation.01'] = (kde, b)
        # f.create_dataset('kde.01', kde_vals.shape, dtype=kde_vals.dtype)[:] = kde_vals
        f.create_dataset('kde.validation.01', data = kde_vals)
        kde_vals = batch_kde(train, train_lengths, test, test_lengths, b)
        kde = np.median(kde_vals)
        f.create_dataset('kde.test.01', data = kde_vals)
        f.attrs['kde.test.01'] = (kde,b)

        for target in [0.001, 0.0001, 0.00001]:
            kde, kde_vals, b = scan_for_kde(train, validation, b/10, b, target=target)
            print(f"{kde} with b={b} for target={target}")
            ds_str = 'kde.validation' + '{:f}'.format(target).strip('0')
            # f.create_dataset(ds_str, kde_vals.shape, dtype=kde_vals.dtype)[:] = kde_vals
            f.create_dataset(ds_str, data = kde_vals)
            f.attrs[ds_str] = (kde, b)
            ds_str = 'kde.test' + '{:f}'.format(target).strip('0')
            kde_vals = batch_kde(train, train_lengths, test, test_lengths, b)
            kde = np.median(kde_vals)
            f.create_dataset(ds_str, data = kde_vals)
            f.attrs[ds_str] = (kde, b)

        # kde, kde_vals, b = scan_for_kde(data, query, b/10, b, target=0.0001)
        # print(f"{kde} with b={b}")
        # f.create_dataset('kde.0001', kde_vals.shape, dtype=kde_vals.dtype)[:] = kde_vals
        # f.attrs['kde.0001'] = (kde, b)


        print(f"Computing bandwidth took {time.time() - s}s.")

    # if compute_dists:
    #     start = time.time()
    #     dists = np.sqrt(np.maximum((query**2).sum(-1)[:,None] - \
    #                                    2.0*query.dot(data.T) + \
    #                                    (data**2).sum(-1)[None,:], 0.0))
    #     end = time.time()
    #     print(f'distance (l2) computation took {end-start} s')
    #     start = time.time()
    #     f.create_dataset('dists', data=dists)
    #     end = time.time()
    #     print(f'storing of dists (l2) took {end-start} s')

    #     start = time.time()
    #     dists_l1 = np.zeros((query.shape[0], data.shape[0]), np.float64)
    #     for i in range(query.shape[0]):
    #         q = query[i,:]
    #         dists_l1[i,:] = np.abs(q[None,:] - data).sum(-1)
    #     end = time.time()
    #     print(f'distance (l1) computation took {end-start} s')

    #     start = time.time()
    #     f.create_dataset('dists.l1', data=dists_l1)
    #     end = time.time()
    #     print(f'storing of dists (l1) took {end-start} s')


    f.close()


def covtype(out_fn, compute_bw):
    import gzip
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
    fn = os.path.join('data', 'covtype.gz')
    download(url, fn)
    X = []
    with gzip.open(fn, 'rt') as t:
        for line in t.readlines():
            X.append([int(x) for x in line.strip().split(",")])
    write_output(np.array(X), out_fn, compute_bw)


def covtype_preprocess(fn):
    import gzip
    X = []
    with gzip.open(fn, 'rt') as t:
        for line in t.readlines():
            X.append([int(x) for x in line.strip().split(",")][:-1])
    return np.array(X) # ,dtype=np.float64)


def census(out_fn, compute_bw, compute_nn):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
    fn = os.path.join('data', 'census.txt')
    download(url, fn)
    X = []
    with open(fn) as f:
        # skip headerline
        for line in f.readlines()[1:]:
            X.append(list(map(int, line.split(",")[1:])))
    write_output(np.array(X), out_fn, compute_bw, compute_nn)



def census_preprocess(filename):
    X = []
    with open(filename) as f:
        # skip headerline, drop caseid
        for line in f.readlines()[1:]:
            X.append(list(map(int, line.split(",")[1:])))
    return np.array(X) #, dtype=np.float64)



def shuttle(out_fn, compute_bw):
    import zipfile
    X = []
    for dn in ("shuttle.trn.Z", "shuttle.tst"):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/%s" % dn
        fn = os.path.join("data", dn)
        download(url, fn)
        if fn.endswith(".Z"):
            os.system("uncompress " + fn)
            fn = fn[:-2]
        with open(fn) as f:
            for line in f:
                X.append([int(x) for x in line.split()])
    write_output(np.array(X), out_fn, compute_bw)



def shuttle_preprocess(filenames):
    import zipfile
    X = []
    for fn in filenames:
        if fn.endswith(".Z"):
            if not os.path.exists(fn[:-2]):
                os.system("uncompress " + fn)
            fn = fn[:-2]
        with open(fn) as f:
            for line in f:
                # drop the class label
                X.append([int(x) for x in line.split()][:-1])
    return np.array(X)#,dtype=np.float64)


def glove(out_fn, compute_bw, d=100):
    import zipfile

    url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip'
    fn = os.path.join('data', 'glove.twitter.27B.zip')
    download(url, fn)
    with zipfile.ZipFile(fn) as z:
        print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
        write_output(np.array(X), out_fn, compute_bw)

def glove_preprocess(fn):
    import zipfile
    d=100

    with zipfile.ZipFile(fn) as z:
        # print('preparing %s' % out_fn)
        z_fn = 'glove.twitter.27B.%dd.txt' % d
        X = []
        for line in z.open(z_fn):
            v = [float(x) for x in line.strip().split()[1:]]
            X.append(np.array(v))
    return np.array(X)

def _load_texmex_vectors(f, n, k):
    import struct

    v = np.zeros((n, k))
    for i in range(n):
        f.read(4)  # ignore vec length
        v[i] = struct.unpack('f' * k, f.read(k * 4))

    return v


def _get_irisa_matrix(t, fn):
    import struct
    m = t.getmember(fn)
    f = t.extractfile(m)
    k, = struct.unpack('i', f.read(4))
    n = m.size // (4 + 4 * k)
    f.seek(0)
    return _load_texmex_vectors(f, n, k)


def sift(out_fn, compute_bw):
    import tarfile

    url = 'ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz'
    fn = os.path.join('data', 'sift.tar.tz')
    download(url, fn)
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
        write_output(train, out_fn, compute_bw)

def sift_preprocess(fn):
    import tarfile
    with tarfile.open(fn, 'r:gz') as t:
        train = _get_irisa_matrix(t, 'sift/sift_base.fvecs')
    return train


def svhn(out_fn, compute_bw, version):
    from scipy.io import loadmat
    url = 'http://ufldl.stanford.edu/housenumbers/%s_32x32.mat' % version
    fn = os.path.join('data', 'svhn-%s.mat' % version)
    download(url, fn)
    X = loadmat(fn)['X']
    d = np.prod(X.shape[:3])
    Y = np.reshape(X, (d, X.shape[3])).T
    write_output(Y, out_fn, compute_bw)

def svhn_preprocess(fn):
    from scipy.io import loadmat
    X = loadmat(fn)['X']
    d = np.prod(X.shape[:3])
    Y = np.reshape(X, (d, X.shape[3])).T
    return Y # .astype(np.float32)


def _load_mnist_vectors(fn):
    import gzip
    import struct

    print('parsing vectors in %s...' % fn)
    f = gzip.open(fn)
    type_code_info = {
        0x08: (1, "!B"),
        0x09: (1, "!b"),
        0x0B: (2, "!H"),
        0x0C: (4, "!I"),
        0x0D: (4, "!f"),
        0x0E: (8, "!d")
    }
    magic, type_code, dim_count = struct.unpack("!hBB", f.read(4))
    assert magic == 0
    assert type_code in type_code_info

    dimensions = [struct.unpack("!I", f.read(4))[0]
                  for i in range(dim_count)]

    entry_count = dimensions[0]
    entry_size = np.product(dimensions[1:])

    b, format_string = type_code_info[type_code]
    vectors = []
    for i in range(entry_count):
        vectors.append([struct.unpack(format_string, f.read(b))[0]
                        for j in range(entry_size)])
    return np.array(vectors)


def mnist(out_fn, compute_bw):
    download('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
	os.path.join('data', 'mnist-train.gz'))
    train = _load_mnist_vectors('mnist-train.gz')
    write_output(train, out_fn, compute_bw)

def mnist_preprocess(fn):
    return _load_mnist_vectors(fn)



def fashion_mnist(out_fn, compute_bw):
    download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',  # noqa
             os.path.join('data', 'fashion-mnist-train.gz'))
    #download('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',  # noqa
    #         'fashion-mnist-test.gz')
    train = _load_mnist_vectors('fashion-mnist-train.gz')
    #test = _load_mnist_vectors('fashion-mnist-test.gz')
    #write_output(train, out_fn, compute_bw)

def lastfm(out_fn, compute_bw, n_dimensions=64, test_size=50000):
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    from implicit.datasets.lastfm import get_lastfm
    from implicit.approximate_als import augment_inner_product_matrix
    import implicit

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(
        play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = np.append(model.user_factors,
                                np.zeros((model.user_factors.shape[0], 1)),
                                axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    write_output(item_factors, out_fn, compute_bw, queries=test_size)



def lastfm_preprocess(fn):
    n_dimensions=64
    test_size=50000
    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)
    from implicit.datasets.lastfm import get_lastfm
    from implicit.approximate_als import augment_inner_product_matrix
    import implicit

    # train an als model on the lastfm data
    _, _, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions)
    model.fit(implicit.nearest_neighbours.bm25_weight(
        play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = np.append(model.user_factors,
                                np.zeros((model.user_factors.shape[0], 1)),
                                axis=1)

    # only query the first 50k users (speeds things up signficantly
    # without changing results)
    user_factors = user_factors[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data
    # write_output(item_factors, out_fn, compute_bw, queries=test_size)
    assert isinstance(item_factors, np.ndarray)
    assert item_factors.dtype == np.float32
    return item_factors #.astype(np.float64)


def aloi_preprocess(fn):
    # AR14a
    # Anderson Rocha and Siome Goldenstein.
    # Multiclass from binary: Expanding one-vs-all, one-vs-one and ECOC-based approaches.
    # IEEE Transactions on Neural Networks and Learning Systems, 25(2):289–302, 2014.
    import tarfile
    with tarfile.open(fn, 'r') as f:
        with f.extractfile('aloi.data') as d:
            data = d.read().decode('ascii').strip().split('\n')
            X = [list(map(int,data[i].split(' ')[1:-1])) \
                     for i in range(1,len(data))]
    return np.array(X)


def msd_preprocess(fn):
    import zipfile
    X = list()
    with zipfile.ZipFile(fn,'r') as z:
        with z.open('YearPredictionMSD.txt','r') as f:
            X = [list(map(float, line.decode('ascii').strip().split(',')[1:])) \
                              for line in f]
    return np.array(X)



def timit_preprocess(fn):
    # import zipfile
    # with zipfile.ZipFile(fn,'r') as z:
    #     with z.open('test_data.csv','r') as f:
    #         for line in f:
    #             print(line)
    #             break
    # apparently, this dataset is preprocessed by using a neural network
    # autoencoder which truncates individual files into 440-long vectors
    # I am unable to find the plain dataset in the wild
    pass


def cadata_preprocess(fn):
    import zipfile
    X = list()
    with zipfile.ZipFile(fn,'r') as z:
        with z.open('cadata.txt','r') as f:
            data = f.readlines()
    X = [list(map(float,
     (line.decode('ascii')[25*i:25*(i+1)] for i in range(1,9))))
             for line in data[27:]]
    return np.array(X)


def poker_preprocess(fns):
    X = list()
    for fn in fns:
        with open(fn,'r') as f:
            for line in f:
                X.append(list(map(int,line.strip().split(',')[:-1])))
    return np.array(X)



def codrna_preprocess(fns):
    X = []
    for fn in fns:
        with open(fn,'r') as f:
            for line in f:
                X.append(list(map(float, map(lambda x: x[2:], line.lstrip('-1').strip().split(' ')))))
    return np.array(X)



def sensorless_preprocess(fn):
    with open(fn,'r') as f:
        data = [list(map(float,line.strip().split(' ')[:-1])) for line in f]
    X = np.array(data)
    return X



def corel_preprocess(fn):
    import gzip
    with gzip.open(fn, 'rt') as f:
        data = [list(map(float,line.split(' ')[1:])) for line in f]
    X = np.array(data)
    return X



def acoustic_preprocess(fns):
    import bz2
    data = list()
    for fn in fns:
        with bz2.open(fn,'rt') as f:
            for line in f:
                data.append(list(map(lambda x: float(x[x.find(':')+1:]),
                                    line.strip().split(' ')[1:])))
    X = np.array(data)
    return X



def ijcnn_preprocess(fns):
    import bz2
    data = list()
    for fn in fns:
        with bz2.open(fn,'rt') as f:
            for line in f:
                x = [0.0]*22
                if line.startswith('-1 '):
                    line = line[3:]
                elif line.startswith('1 '):
                    line = line[2:]
                elif line.startswith('-1.0 '):
                    line = line[5:]
                elif line.startswith('1.0 '):
                    line = line[4:]
                else:
                    print(fn)
                    assert False
                for field in line.split(' '):
                    field = field.split(':')
                    idx = int(field[0])
                    val = float(field[1])
                    x[idx-1] = val
                data.append(x)
    X = np.array(data)
    return X



def skin_preprocess(fn):
    with open(fn,'r') as f:
        data = [list(map(int,line.split('\t')[:3])) for line in f]
    return np.array(data)



def home_preprocess(fn):
    import zipfile
    with zipfile.ZipFile(fn,'r') as z:
        with z.open('HT_Sensor_dataset.zip','r') as f:
            with zipfile.ZipFile(f,'r') as y:
                with y.open('HT_Sensor_dataset.dat','r') as g:
                    data = g.readlines()
    data = [list(map(float,line.decode('ascii').strip().split('  ')[2:])) \
                for line in data[1:]]
    X = np.array(data)
    return X



def susy_preprocess(fn):
    import gzip
    with gzip.open(fn,'rt') as f:
        data = [list(map(float,line.strip().split(',')[1:])) for line in f]
    X = np.array(data)
    return X



def hep_preprocess(fns):
    import gzip
    data = list()
    for fn in fns:
        with gzip.open(fn,'rt') as f:
            for line in f:
                if line.startswith('# label'):
                    continue
                data.append(list(map(float,line.strip().split(',')[1:-1])))
    X = np.array(data)
    return X



def higgs_preprocess(fn):
    import gzip
    with gzip.open(fn,'rt') as f:
        data = [list(map(float,line.strip().split(',')[1:])) for line in f]
    X = np.array(data)
    return X


# DATASETS = {
#     'fashion-mnist-784-euclidean': fashion_mnist,
#     'glove': glove,
#     'mnist': mnist,
#     'sift': sift,
#     'lastfm': lastfm,
#     'covtype': covtype,
#     'census': census,
#     'shuttle': shuttle,
#     'svhn': lambda out_fn, compute_bw: svhn(out_fn, compute_bw, 'extra'),
#     'svhn-small': lambda out_fn, compute_bw: svhn(out_fn, compute_bw, 'test'),
# }

# url, filename_prefix, raw_filename, preprocess_function
DATASETS = {
    'glove' : ('http://nlp.stanford.edu/data/glove.twitter.27B.zip', 'glove',
                   'glove.twitter.27B.zip', glove_preprocess),
    'mnist' : ('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
                   'mnist', 'mnist-train.gz', mnist_preprocess),
    'sift' : ('ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz', 'sift',
                  'sift.tar.tz', sift_preprocess),
    'lastfm' : (None, 'lastfm', None, lastfm_preprocess),
    'covtype' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz',
                     'covtype', 'covtype.gz', covtype_preprocess),
    'census' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt',
                'census', 'census.txt', census_preprocess),
    'shuttle' : (list("https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/shuttle/%s" % dn \
                          for dn in ("shuttle.trn.Z", "shuttle.tst")),
                 'shuttle', ["shuttle.trn.Z", "shuttle.tst"], shuttle_preprocess),
    'svhn-small' : ('http://ufldl.stanford.edu/housenumbers/test_32x32.mat',
                        'svhn-small', 'svhn-test.mat', svhn_preprocess),
    'svhn' : ('http://ufldl.stanford.edu/housenumbers/extra_32x32.mat',
                  'svhn', 'svhn-extra.mat', svhn_preprocess),
    'aloi' : ('https://ic.unicamp.br/~rocha/pub/downloads/2014-tnnls/aloi.tar.gz',
                  'aloi', 'aloi.tar.gz', aloi_preprocess),
    'msd'  : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
                  'msd', 'YearPredictionMSD.txt.zip', msd_preprocess),
    # 'timit' : ('https://data.deepai.org/timit.zip', 'timit', 'timit.zip',
    #                timit_preprocess),
    # also tmy3 could not be found
    'cadata' : ('http://lib.stat.cmu.edu/datasets/houses.zip', 'cadata',
                    'houses.zip', cadata_preprocess),
    'poker' : (['https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-testing.data',
                    'https://archive.ics.uci.edu/ml/machine-learning-databases/poker/poker-hand-training-true.data'],
                    'poker', ['poker-hand-testing.data',
                                  'poker-hand-training-true.data'],
                                  poker_preprocess),
    'codrna' : (['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna',
     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.t',
     'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna.r'],
     'codrna', ['cod-rna', 'cod-rna.t', 'cod-rna.r'], codrna_preprocess),
    'sensorless' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00325/Sensorless_drive_diagnosis.txt',
                         'sensorless', 'Sensorless_drive_diagnosis.txt',
                         sensorless_preprocess),
    'corel' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/CorelFeatures-mld/ColorHistogram.asc.gz',
                  'corel', 'ColorHistogram.asc.gz', corel_preprocess),
    'acoustic' : (['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic.bz2',
                       'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/vehicle/acoustic.t.bz2'],
                      'acoustic', ['acoustic.bz2', 'acoustic.t.bz2'],
                      acoustic_preprocess),
    'ijcnn' : (['https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.bz2',
                    'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2',
                    'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2',
                    'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2'],
                    'ijcnn', ['ijcnn1.bz2', 'ijcnn1.t.bz2', 'ijcnn1.tr.bz2', 'ijcnn1.val.bz2'],
                    ijcnn_preprocess),
    'skin' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt',
                  'skin', 'Skin_NonSkin.txt', skin_preprocess),
    'home' : ('http://archive.ics.uci.edu/ml/machine-learning-databases/00362/HT_Sensor_UCIsubmission.zip',
                  'home', 'HT_Sensor_UCIsubmission.zip', home_preprocess),
    'susy' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz',
                  'susy', 'SUSY.csv.gz', susy_preprocess),
    'hep' : (['http://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_test.csv.gz',
                  'http://archive.ics.uci.edu/ml/machine-learning-databases/00347/all_train.csv.gz'],
                 'hep', ['all_test.csv.gz', 'all_train.csv.gz'],
                 hep_preprocess),
    'higgs' : ('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz',
                   'higgs', 'HIGGS.csv.gz', higgs_preprocess),
}



def create_dataset(dataset, compute_bandwidth=True): #, compute_nn=False, compute_dists=False):
    fn = get_dataset_fn(dataset)

    if not os.path.exists('data'):
        os.mkdir('data')
    (url, filename_prefix, raw_filename, preprocess_function) = DATASETS[dataset]
    if raw_filename is None:
        download_filename = None
    elif isinstance(url,list):
        assert isinstance(raw_filename,list)
        assert len(url) == len(raw_filename)
        download_filename = list()
        for (u,dfn) in zip(url,map(lambda f: f'data/{f}', raw_filename)):
            download(u,dfn)
            download_filename.append(dfn)
    else:
        assert isinstance(raw_filename,str)
        download_filename = f'data/{raw_filename}'
        download(url, download_filename)

    start = time.time()
    X = preprocess_function(download_filename).astype(np.float64)
    end = time.time()
    print(f'data preprocessing took {end-start} s')

    output_filename = f'data/{filename_prefix}.hdf5'

    start = time.time()
    queries=500
    # data, query = train_test_split(X, test_size=queries, random_state=42)
    train, test = train_test_split(X, test_size=2*queries, random_state=42)
    validation, test = train_test_split(test, test_size=queries, random_state=42)
    end = time.time()
    print(f'train test split took {end-start} s')

    # free memory
    del X

    write_output(train, validation, test, output_filename, compute_bandwidth)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        choices=DATASETS.keys(),
        required=True)
    parser.add_argument(
        '--compute-bandwidth',
        action="store_true"
    )
    # parser.add_argument(
    #     '--compute-nn',
    #     action="store_true"
    # )
    # parser.add_argument(
    #     '--compute-dists',
    #     action="store_true"
    # )
    args = parser.parse_args()
    #DATASETS[args.dataset](fn, args.compute_bandwidth, args.compute_nn)
    create_dataset(args.dataset, args.compute_bandwidth)



if __name__ == "__main__":
    main()


