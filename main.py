import argparse

import algorithms
import matrix_game
import time
import traceback
import utils

from algorithms import *

parser = argparse.ArgumentParser()
parser.add_argument('--feedback', default=FULL_FEEDBACK, type=str, choices=(FULL_FEEDBACK, NOISY_FEEDBACK), help="feedback type")
parser.add_argument('--noise_scale', default=0.1, type=float, choices=(0.01, 0.1), help='gaussian noise scale with noisy feedback')
parser.add_argument('--payoff', default='payoff/baised_rsp.csv', type=str, help='path to payoff csv file')
parser.add_argument('--n_p', '--n_processes', type=int, default=1, help="number of process different seed")
parser.add_argument('--n_i', '--n_iterations', type=int, default=10000, help='number of iterations')
parser.add_argument('--update_freq', '--uf', type=int, default=0, help='updating frequence of reference strategy. update_freq=0 ')
parser.add_argument('--arch', type=str, default='m2wu',  choices=['m2wu', 'mwu', 'omwu'])
parser.add_argument('--mu', type=float, default=0.1, help='mutation parameter')
parser.add_argument('--eta', type=float, default=0.1, help='learning rate')
parser.add_argument('--outdir', type=str, default='results', help='Directory path to save output files. If it does not exist, it will be created.')
parser.add_argument('--dir_name', type=str, default='', help='suffix of result dirpath')
parser.add_argument('--seed', type=int, default=0, help="random seed")
parser.add_argument('--r_p', '--random_payoff', action='store_true', help='whether to use random payoff')
parser.add_argument('--r_i_s', '--random_init_strategy', action='store_true', help='whether to generate the initial strategy uniformly at random')
parser.add_argument('--decay' , action='store_true', help='whether to decay learning rate')
parser.add_argument('--size', type=int, default=25, choices=[25, 100], help='random payoff size')


def main(args):
    if args.r_p:
        payoffs = utils.load_payoff_all_arrays('payoff/gaussian_size_{}'.format(args.size))
    else:
        payoff = utils.load_payoff_array(args.payoff)
        payoffs = [payoff] * args.n_p

    inputList = [(p_id, args, payoffs[p_id], np.random.randint(0, 2 ** 32)) for p_id in range(args.n_p)]
    n_pool = min(args.n_p, int(utils.get_cpu_count()-1))
    utils.run_async_pool(n_pool, run_mwu, inputList)


def run_mwu(inputs):
    try:
        process_idx, args, payoff, seed = inputs
        utils.set_random_seed(seed)
        utils.save_payoff_array('{}/csv/seed_{}_payoff.csv'.format(args.outdir, process_idx), payoff)
        game = matrix_game.MatrixGame(payoff, guassian_scale=args.noise_scale)

        n_actions_tuple = game.n_actions_tuple()
        if args.arch == 'mwu':
            agents = [
                MWU(args.eta, n_actions_tuple[0], decay=args.decay),
                MWU(args.eta, n_actions_tuple[1], decay=args.decay)
                ]
        elif args.arch == 'omwu':
            agents = [
                OMWU(args.eta, n_actions_tuple[0], decay=args.decay),
                OMWU(args.eta, n_actions_tuple[1], decay=args.decay)
                ]
        elif args.arch == 'm2wu':
            agents = [
                M2WU(args.eta, n_actions_tuple[0], args.mu, update_freq=args.update_freq, decay=args.decay),
                M2WU(args.eta, n_actions_tuple[1], args.mu, update_freq=args.update_freq, decay=args.decay),
                ]
        else:
            assert False

        algorithms.run_mwu(
            p_id=process_idx,
            game=game,
            agents=agents,
            n_iterations=args.n_i,
            outdir=args.outdir,
            feedback=args.feedback,
            random_strategy=args.r_i_s
        )

        print('Finish process id {}'.format(process_idx))
    except Exception as e:
        traceback.print_exc()
        print(e)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.r_p:
        payoff = 'payoff/gaussian_size_{}'.format(args.size)
    else:
        payoff = args.payoff
        payoff = payoff.replace('.csv', '')
    payoff = payoff.replace('/', '_')
    args.dir_name = 'arch_{}_{}_feedback_{}_{}'.format(args.arch, payoff, args.feedback, args.dir_name) \
        if args.dir_name != '' else 'arch_{}_{}_feedback_{}'.format(args.arch, payoff, args.feedback)
    args.outdir = utils.prepare_output_dir(args, args.outdir)
    utils.set_random_seed(args.seed)
    start = time.time()
    main(args)
    elapsed_time = time.time() - start
    print(args.outdir, " elapsed_time:{0}".format(elapsed_time) + "[sec]")
