from result import get_all_results

def filter_hbe(dataset, query_set, mu, run):
    eps, tau = run
    d = {}
    for f in get_all_results(dataset, "hbe", query_set, mu):
        if 'err' in f.attrs:
            params = f.attrs['params']
            params = params.split("_")[1][1:-1]
            _eps, _tau = map(float, params.split(","))
            if _tau not in d:
                d[_tau] = _eps
            d[_tau] = max(d[_tau], _eps)
    print(d)
    for _tau, _eps in d.items():
        if _tau <= tau and eps <= _eps:
            return True # skip run because we have seen a 'faster run' that timed out
    return False


def filter_run(algo, dataset, query_set, mu, run):
    if algo == "hbe":
        return filter_hbe(dataset, query_set, mu, run)
    return False

    