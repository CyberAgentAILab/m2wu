import multiprocessing as mp
import numpy as np


def set_random_seed(seed):
    np.random.seed(seed)


def load_utility_matrix(game, id):
    if 'random_utility' in game:
        return np.loadtxt('utility/{}/utility{}.csv'.format(game, id), delimiter=',')
    else:
        return np.loadtxt('utility/{}.csv'.format(game), delimiter=',')


def run_async_pool(n_process, run_func, inputs_list):
    multi_pool = mp.Pool(n_process)
    for _ in multi_pool.imap_unordered(run_func, inputs_list):
        pass


def get_cpu_count():
    return mp.cpu_count()


def save_utility_matrix(file_name, utility):
    np.savetxt(file_name, utility, fmt='%.8f', delimiter=',')
