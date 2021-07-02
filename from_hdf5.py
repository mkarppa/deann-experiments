import h5py
import os
import sys
import numpy as np


def check_dirs(ds_name):
    for d in ['external', 'external/rehashing', 'external/rehashing/resources',
                  'external/rehashing/resources/data']:
        if not os.path.isdir(d):
            raise NotADirectoryError(d)


def get_dataset_fn(dataset):
    if not os.path.exists('external/rehashing/resources/data'):
        os.mkdir('external/rehashing/resources/data')
    if not os.path.exists('external/rehashing/resources/queries'):
        os.mkdir('external/rehashing/resources/queries')
    if not os.path.exists('external/rehashing/resources/exact'):
        os.mkdir('external/rehashing/resources/exact')
    return (os.path.join('external/rehashing/resources', 'data', '%s.txt' % dataset),
            os.path.join('external/rehashing/resources', 'queries', '%s.txt' % dataset),
            os.path.join('external/rehashing/resources', 'data', '%s.conf' % dataset),
            os.path.join('external/rehashing/resources', 'exact', '%s_exp.txt' % dataset),
            )


# def write_data(ds_name, dataset, X):

    
# def write_data(name, data, queries, ground_truth):
#     data_fn, query_fn, _, groundtruth_fn = get_dataset_fn(name)
#     print(get_dataset_fn(name))
#     with open(data_fn, 'w') as f:
#         for i, v in enumerate(data):
#             f.write(str(i) + "," + ",".join(map(str, v)) + "\n")
#     with open(query_fn, 'w') as f:
#         for i, v in enumerate(queries):
#             f.write(str(i) + "," + ",".join(map(str, v)) + "\n")
#     with open(groundtruth_fn, 'w') as f:
#         for i, val in enumerate(ground_truth):
#             f.write(f'{val},{i}\n')


def write_config(name, n, d, m, bw):
    data_fn, query_fn, config_fn, groundtruth_fn = get_dataset_fn(name)
    print(f'writing config to {config_fn}')
    f = open(config_fn, 'w')

    config = """
    exp {
        name = "%s";
        fpath = "%s";
        qpath = "%s";
        exact_path = "%s";
        kernel = "exp";
        n = "%d";
        d = "%d";
        m = "%d";
        h = "%d";
        bw_const = "true";
        ignore_header = "false";
        start_col = "1";
        end_col = "%d";
        samples = "100";
        sample_ratio = "2.5";
        eps = "0.5";
        tau = "0.000001";
        beta = "0.5";
    }
    """ % (name, data_fn, query_fn, groundtruth_fn,
            n, d, m, bw, d)
    f.write(config)
    f.close()


    
def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file>")
        exit(1)

    fn = sys.argv[1]
    ds_name = fn.split("/")[-1][:-5]
    check_dirs(ds_name)

    with h5py.File(fn, 'r') as f:
        for dataset in ['train', 'validation', 'test',
                            'kde.validation.01', 'kde.validation.001',
                            'kde.validation.0001', 'kde.validation.00001',
                            'kde.test.01', 'kde.test.001',
                            'kde.test.0001', 'kde.test.00001']:
            np.savetxt(f'external/rehashing/resources/data/{ds_name}' +
                           f'.{dataset}.csv',
                           np.array(f[dataset]), delimiter=',')

    

if __name__ == "__main__":
    main()











