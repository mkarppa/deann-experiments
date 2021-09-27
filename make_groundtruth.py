import argparse
import logging
import time
import resource
import pdb

import numpy as np

import faiss

from faiss.contrib.exhaustive_search import range_search_gpu

from preprocess_datasets import get_dataset,DATASETS

"""
for dataset in shuttle; do
    sbatch   --gres=gpu:4 --ntasks=1 --time=30:00:00 --cpus-per-task=40        \
           --partition=learnlab --mem=250g --nodes=1  \
           -J GT.100M.$dataset.d -o logs/GT.100M.$dataset.d.log \
           --wrap "PYTHONPATH=. python make_groundtruth.py \
            --dataset $dataset \
            --o /home/maau/${dataset}
        "
done
"""


class ResultHeap:
    """Accumulate query results from a sliced dataset. The final result will
    be in self.D, self.I."""

    def __init__(self, nq, k, keep_max=False):
        " nq: number of query vectors, k: number of results per query "
        self.I = np.zeros((nq, k), dtype='int64')
        self.D = np.zeros((nq, k), dtype='float32')
        self.nq, self.k = nq, k
        if keep_max:
            heaps = faiss.float_minheap_array_t()
        else:
            heaps = faiss.float_maxheap_array_t()
        heaps.k = k
        heaps.nh = nq
        heaps.val = faiss.swig_ptr(self.D)
        heaps.ids = faiss.swig_ptr(self.I)
        heaps.heapify()
        self.heaps = heaps

    def add_result(self, D, I):
        """D, I do not need to be in a particular order (heap or sorted)"""
        assert D.shape == (self.nq, self.k)
        assert I.shape == (self.nq, self.k)
        self.heaps.addn_with_ids(
            self.k, faiss.swig_ptr(D),
            faiss.swig_ptr(I), self.k)

    def finalize(self):
        self.heaps.reorder()


def knn_ground_truth(ds, k):
    """Computes the exact KNN search results for a dataset that possibly
    does not fit in RAM but for which we have an iterator that
    returns it block by block.
    """
    print("loading queries")
    xq = ds.get_queries()

    print("knn_ground_truth queries size %s k=%d" % (xq.shape, k))

    t0 = time.time()
    nq, d = xq.shape

    rh = ResultHeap(nq, k, keep_max=False)

    index = faiss.IndexFlat(d, faiss.METRIC_L2)

    if faiss.get_num_gpus():
        print('running on %d GPUs' % faiss.get_num_gpus())
        index = faiss.index_cpu_to_all_gpus(index)

    # compute ground-truth by blocks, and add to heaps
    index.add(ds['train'])
    D, I = index.search(ds['train'], k)
    print()
    print("GT time: %.3f s (%d vectors)" % (time.time() - t0, i0))

    return D, I


def usbin_write(ids, dist, fname):
    ids = np.ascontiguousarray(ids, dtype="int32")
    dist = np.ascontiguousarray(dist, dtype="float32")
    assert ids.shape == dist.shape
    f = open(fname, "wb")
    n, d = dist.shape
    np.array([n, d], dtype='uint32').tofile(f)
    ids.tofile(f)
    dist.tofile(f)


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
    aa("--maxRAM", default=100, type=int, help="set max RSS in GB (avoid OOM crash)")

    group = parser.add_argument_group('output options')
    aa('--o', default="", help="output file name")

    args = parser.parse_args()

    print("args:", args)

    if args.maxRAM > 0:
        print("setting max RSS to", args.maxRAM, "GiB")
        resource.setrlimit(
            resource.RLIMIT_DATA, (args.maxRAM * 1024 ** 3, resource.RLIM_INFINITY)
        )

    ds = get_dataset(args.dataset)

    D, I = knn_ground_truth(ds, k=args.k)
    print(f"writing index matrix of size {I.shape} to {args.o}")
    # write in the usbin format
    usbin_write(I, D, args.o)
