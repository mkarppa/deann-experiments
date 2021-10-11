from result import get_all_results
import h5py

def filter_hbe(dataset, query_set, mu, run, algo = 'hbe'):
    eps, tau = run
    d = {}
    for f in get_all_results(dataset, algo, query_set, mu):
        if 'err' in f.attrs:
            params = f.attrs['params']
            params = params.split("_")[1][1:-1]
            _eps, _tau = map(float, params.split(","))
            if _tau not in d:
                d[_tau] = _eps
            d[_tau] = max(d[_tau], _eps)
    print(d)
    # print(tau,eps)
    for _tau, _eps in d.items():
        if tau <= _tau and eps <= _eps:
            # print('skipping ({},{})'.format(tau,eps))
            return True # skip run because we have seen a 'faster run' that timed out
    # print('not skipping ({},{})'.format(tau,eps))
    return False



def filter_sklearn(dataset, query_set, mu, run, algo):
    (l,_,rtol) = run
    d = {}
    for f in get_all_results(dataset, algo, query_set, mu):
        if 'err' in f.attrs:
            params = f.attrs['params']
            params = params.split("_")[1][1:-1]
            params = params.split(',')
            _l = int(params[0])
            _rtol = float(params[2])
            if _l not in d:
                d[_l] = _rtol
            d[_l] = max(d[_l],_rtol)
    print(d)
    # print(tau,eps)
    # for _tau, _eps in d.items():
    #     if tau <= _tau and eps <= _eps:
    #         # print('skipping ({},{})'.format(tau,eps))
    #         return True # skip run because we have seen a 'faster run' that timed out
    # print('not skipping ({},{})'.format(tau,eps))
    if l in d and rtol < d[l]:
        return True
    else:
        return False


def filter_run(algo, dataset, query_set, mu, run):
    if algo in ["hbe", 'rs']:
        return filter_hbe(dataset, query_set, mu, run, algo)
    elif algo in ['sklearn-kdtree', 'sklearn-balltree']:
        return filter_sklearn(dataset, query_set, mu, run, algo)
        
    return False

def filter_runs_deann(algo, dataset, query_set, mu, query_args):
    assert algo in ['naive', 'rsp', 'random-sampling', 'ann-faiss', 'ann-permuted-faiss']
    if algo in ['naive', 'rsp', 'ann-permuted-faiss']:
        return query_args
    with h5py.File(f'data/{dataset}.hdf5','r') as f:
        n,d = f['train'].shape

    new_args = list()
    if algo == 'ann-faiss':
        for (k,m,nl,nq) in query_args:
            if k+m > n:
                continue
            else:
                new_args.append([k,m,nl,nq])
    else:
        assert algo == 'random-sampling'
        for m in query_args:
            if m > n:
                continue
            else:
                new_args.append(m)
    return new_args
