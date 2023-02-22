import numpy as np

from games.matrix_game import MatrixGame
from runner import utils
from runner.logger import Logger


def run_mwu(trial_id, game, T, seed, feedback, alg, params):
    # sed random seed
    utils.set_random_seed(seed)

    # initialize game and players
    game = MatrixGame(utils.load_utility_matrix(game, trial_id))
    players = [
        alg(game.num_actions(0), **params),
        alg(game.num_actions(1), **params)
    ]

    # run each trial
    logger = Logger()
    for t in np.arange(T + 1):
        if feedback == 'full':
            strategies = [agent.strategy for agent in players]
            utilities = game.full_feedback(strategies)
            for i_a, agent in enumerate(players):
                agent.update(utilities[i_a])
        elif feedback == 'noisy':
            strategies = [agent.strategy for agent in players]
            utilities = game.noisy_feedback(strategies)
            for i_a, agent in enumerate(players):
                agent.update(utilities[i_a])
        else:
            raise RuntimeError('illegal feedback type')
        exploitability = game.calc_exploitability(strategies)
        if t < 100 or t % max(1, T / int(1e6)) == 0:
            logger['iteration'].append(t)
            logger['exploitability'].append(exploitability)
        if t % 10000 == 0:
            print('trial: {}, iteration: {}, exploitability: {}'.format(trial_id, t, exploitability))
    return logger, game.utility
