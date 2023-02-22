import argparse
import numpy as np
import os

from algorithms import *
from concurrent.futures import ProcessPoolExecutor
from runner import utils
from runner.runner import run_mwu


def run_exp(num_trials, game, T, seed, feedback, alg, params):
    print('==========Run experiments for {}=========='.format(alg.name(params)))
    dir_name = 'results/{}_feedback/{}/{}'.format(feedback, game, alg.name(params))
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        os.makedirs(dir_name + '/csv')
        os.makedirs(dir_name + '/figure')

    # set random seed
    utils.set_random_seed(seed)

    # run trials
    logs = [None] * num_trials
    with ProcessPoolExecutor() as pool:
        arguments = [[trial_id, game, T, np.random.randint(0, 2 ** 32), feedback, alg, params, dir_name] for trial_id in range(num_trials)]
        for trial_id, log in enumerate(pool.map(run_mwu, *tuple(zip(*arguments)))):
            logs[trial_id] = log

    # save log
    utils.save_and_summary_results(dir_name, logs, 'log' if feedback == 'full' else 'linear', 'log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='biased_rps', type=str,
                        choices=['biased_rps', 'm_ne', *['random_utility/size{}'.format(s) for s in [25, 100]]], help='name of game')
    parser.add_argument('--num_trials', type=int, default=1, help="number of trials to run experiments")
    parser.add_argument('--T', type=int, default=10000, help='number of iterations')
    parser.add_argument('--feedback', type=str, default='full', choices=['full', 'noisy'], help="feedback type")
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--algorithm', type=str, default='m2wu',  choices=['m2wu', 'mwu', 'omwu'])
    parser.add_argument('--random_init_strategy', action='store_true', help='whether to generate the initial strategy uniformly at random')
    parser.add_argument('--eta', type=float, default=0.1, help='learning rate')
    parser.add_argument('--decay', action='store_true', help='whether to decay learning rate')
    parser.add_argument('--mu', type=float, default=0.1, help='mutation parameter')
    parser.add_argument('--update_freq', type=int, default=None, help='updating frequence of reference strategy. update_freq=0 ')
    args = parser.parse_args()

    if args.algorithm == 'mwu':
        alg = MWU
        params = {'random_init_strategy': args.random_init_strategy, 'eta': args.eta, 'decay': args.decay}
    elif args.algorithm == 'omwu':
        alg = OMWU
        params = {'random_init_strategy': args.random_init_strategy, 'eta': args.eta, 'decay': args.decay}
    elif args.algorithm == 'm2wu':
        alg = M2WU
        params = {'random_init_strategy': args.random_init_strategy, 'eta': args.eta, 'mu': args.mu, 'update_freq': args.update_freq, 'decay': args.decay}
    else:
        raise RuntimeError('illegal algorithm name')

    # run experiments
    print('==========Run experiment over {} trials=========='.format(args.num_trials))
    run_exp(args.num_trials, args.game, args.T, args.seed, args.feedback, alg, params)


if __name__ == "__main__":
    main()
