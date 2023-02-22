import argparse
import numpy as np
import os

from algorithms import *
from games import matrix_game
from runner import utils
from runner.runner import run_mwu


def run_exp(inputs):
    try:
        process_idx, args, utility, seed, outdir = inputs
        utils.set_random_seed(seed)
        utils.save_utility_matrix('{}/csv/seed_{}_utility.csv'.format(outdir, process_idx), utility)
        game = matrix_game.MatrixGame(utility)

        if args.algorithm == 'mwu':
            agents = [
                MWU(args.eta, game.num_actions(0), decay=args.decay),
                MWU(args.eta, game.num_actions(1), decay=args.decay)
                ]
        elif args.algorithm == 'omwu':
            agents = [
                OMWU(args.eta, game.num_actions(0), decay=args.decay),
                OMWU(args.eta, game.num_actions(1), decay=args.decay)
                ]
        elif args.algorithm == 'm2wu':
            agents = [
                M2WU(args.eta, game.num_actions(0), args.mu, update_freq=args.update_freq, decay=args.decay),
                M2WU(args.eta, game.num_actions(1), args.mu, update_freq=args.update_freq, decay=args.decay),
                ]
        else:
            assert False

        run_mwu(
            p_id=process_idx,
            game=game,
            agents=agents,
            n_iterations=args.T,
            outdir=outdir,
            feedback=args.feedback,
            random_strategy=args.random_init_strategy
        )

        print('Finish process id {}'.format(process_idx))
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--game', default='biased_rps', type=str,
                        choices=['biased_rps', 'm_eq', *['random_utility/size{}'.format(s) for s in [25, 100]]], help='name of game')
    parser.add_argument('--num_trials', type=int, default=1, help="number of trials to run experiments")
    parser.add_argument('--T', type=int, default=10000, help='number of iterations')
    parser.add_argument('--feedback', type=str, default='full', choices=['full', 'noisy'], help="feedback type")
    parser.add_argument('--update_freq', '--uf', type=int, default=0, help='updating frequence of reference strategy. update_freq=0 ')
    parser.add_argument('--algorithm', type=str, default='m2wu',  choices=['m2wu', 'mwu', 'omwu'])
    parser.add_argument('--mu', type=float, default=0.1, help='mutation parameter')
    parser.add_argument('--eta', type=float, default=0.1, help='learning rate')
    parser.add_argument('--dir_name', type=str, default='', help='suffix of result dirpath')
    parser.add_argument('--seed', type=int, default=0, help="random seed")
    parser.add_argument('--random_init_strategy', action='store_true', help='whether to generate the initial strategy uniformly at random')
    parser.add_argument('--decay', action='store_true', help='whether to decay learning rate')
    args = parser.parse_args()

    dir_name = 'results/{}_feedback/{}/{}'.format(args.feedback, args.game, args.algorithm)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        os.makedirs(dir_name + '/csv')
        os.makedirs(dir_name + '/figure')
    utils.set_random_seed(args.seed)
    utilities = [utils.load_utility_matrix(args.game, i) for i in range(args.num_trials)]

    inputList = [(p_id, args, utilities[p_id], np.random.randint(0, 2 ** 32), dir_name) for p_id in range(args.num_trials)]
    n_pool = min(args.num_trials, int(utils.get_cpu_count() - 1))
    utils.run_async_pool(n_pool, run_exp, inputList)


if __name__ == "__main__":
    main()
