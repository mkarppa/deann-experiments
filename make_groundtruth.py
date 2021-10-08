import argparse
import logging
import time
import resource

import numpy as np

import faiss

from preprocess_datasets import DATASETS, get_dataset


def knn_ground_truth(X, k):
    print("knn_ground_truth queries size %s k=%d" % (X.shape, k))

    t0 = time.time()
    _, d = X.shape

    index = faiss.IndexFlat(d, faiss.METRIC_L2)

    index.add(X)
    index.train(X)
    D, I = index.search(X, k)

    return D, I


def usbin_write(ids, dist, fname):
    ids = np.ascontiguousarray(ids, dtype="int64")
    dist = np.ascontiguousarray(dist, dtype="float64")
    assert ids.shape == dist.shape
    f = open(fname, "wb")
    n, d = dist.shape
    np.array([n, d], dtype='uint32').tofile(f)
    ids.tofile(f)
    dist.tofile(f)

def write_data(X, fname):
    f = open(fname, "wb")
    X = np.ascontiguousarray(X, dtype="float64")
    X.tofile(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('dataset options')
    aa('--dataset', choices=DATASETS.keys(), required=True)

    group = parser.add_argument_group('computation options')
    # determined from ds
    # aa('--range_search', action="store_true", help="do range search instead of kNN search")
    aa('--k', default=100, type=int, help="number of nearest kNN neighbors to search")

    args = parser.parse_args()


    ds = get_dataset(args.dataset, "gaussian")

    print(ds)

    for query_type in ('train', 'test', 'validation'):
        D, I = knn_ground_truth(np.array(ds[query_type]).astype(np.float32), k=args.k)
        print(f"writing index matrix of size {I.shape}")
        # write in the usbin format
        usbin_write(I, D, args.dataset + f".{query_type}.knn")
        write_data(np.array(ds[query_type]).astype(np.float64), args.dataset + f".{query_type}.data")


