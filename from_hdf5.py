import h5py
import os
import sys
import numpy as np


def create_dirs():
    hbe_path = os.environ['HBE']
    if not os.path.exists(os.path.join(os.sep, hbe_path, 'resources', 'data')):
        os.mkdir(os.path.join(os.sep, hbe_path, 'resources', 'data'))
    if not os.path.exists(os.path.join(os.sep, hbe_path, 'resources', 'queries')):
        os.mkdir(os.path.join(os.sep, hbe_path, 'resources', 'queries'))
    if not os.path.exists(os.path.join(os.sep, hbe_path, 'resources', 'exact')):
        os.mkdir(os.path.join(os.sep, hbe_path, 'resources', 'exact'))


def get_dataset_fn(dataset):
    hbe_path = os.path.join(os.environ['HBE'], 'resources')
    return (os.path.join(hbe_path, 'data', '%s.txt' % dataset),
            os.path.join(hbe_path, 'queries', '%s.txt' % dataset),
            os.path.join(hbe_path, 'data', '%s.conf' % dataset),
            os.path.join(hbe_path, 'exact', '%s_exp.txt' % dataset),
            )


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


def create_dataset(fn, kernel):
    if 'HBE' not in os.environ:
        print('Error: Please set environmental variable HBE to point to your HBE installation')
        exit(1)
    ds_name = fn.split("/")[-1][:-5]
    create_dirs()

    with h5py.File(fn, 'r') as f:
        for dataset in ['train', 'validation', 'test',
                            f'kde.validation.{kernel}.01',
                            f'kde.validation.{kernel}.001',
                            f'kde.validation.{kernel}.0001',
                            f'kde.validation.{kernel}.00001',
                            f'kde.test.{kernel}.01',
                            f'kde.test.{kernel}.001',
                            f'kde.test.{kernel}.0001',
                            f'kde.test.{kernel}.00001']:
            np.savetxt(f'/{os.environ["HBE"]}/resources/data/{ds_name}' +
                           f'.{dataset}.csv',
                           np.array(f[dataset]), delimiter=',')
    
def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <file>")
        exit(1)

    create_dataset(sys.argv[1])

    

if __name__ == "__main__":
    main()











