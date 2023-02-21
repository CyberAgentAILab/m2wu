import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FULL_FEEDBACK = 'full'
NOISY_FEEDBACK = 'noisy'


def run_mwu(p_id, game, agents, n_iterations, outdir, feedback = FULL_FEEDBACK, random_strategy = False):
    csv_file_path = '{}/csv/seed_{}_exploitability.csv'.format(outdir, p_id)
    png_file_path = '{}/figure/seed_{}_exploitability.png'.format(outdir, p_id)
    exploitabilities = []
    already_write_exploitabilities = False
    index = []
    if random_strategy:
        for i_a, agent in enumerate(agents):
            agent.strategy = random_init_strategy(game.payoff.shape[i_a])
    for i_t in np.arange(0, n_iterations+1):
        if feedback == FULL_FEEDBACK:
            strategies = [agent.strategy for agent in agents]
            utilities = game.full_feedback(strategies)
            for i_a, agent in enumerate(agents):
                agent.update(utilities[i_a])
        elif feedback == NOISY_FEEDBACK:
            strategies = [agent.strategy for agent in agents]
            utilities = game.noisy_feedback(strategies)
            for i_a, agent in enumerate(agents):
                agent.update(utilities[i_a])
        if i_t % 10 == 0 or i_t < 10:
            index.append(i_t)
            exploitabilities.append(game.calc_exploitability(strategies))
        if i_t > 0 and i_t % int(10e5) == 0:
            df = pd.DataFrame(exploitabilities, index=index)
            if already_write_exploitabilities:
                df.to_csv(csv_file_path, mode='a', header=False)
            else:
                df.to_csv(csv_file_path)
            exploitabilities = []
            index = []
            already_write_exploitabilities = True
            if p_id % 10 == 0:
                print('p_id', p_id, ":", i_t, "iterations finished.")
    df = pd.DataFrame(exploitabilities, index=index)
    if already_write_exploitabilities:
        df.to_csv(csv_file_path, mode='a', header=False)
    else:
        df.to_csv(csv_file_path)
    df = pd.read_csv(csv_file_path, index_col=0)
    df.plot()
    plt.title('Exploitability')
    plt.yscale("log")
    plt.savefig(png_file_path)
    plt.clf()
    plt.close()


def random_init_strategy(n_actions):
    random_numbers = np.random.exponential(scale=1.0, size=n_actions)
    return np.array(random_numbers / random_numbers.sum(), dtype=np.float64)