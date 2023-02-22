import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_random_seed(seed):
    np.random.seed(seed)


def load_utility_matrix(game, id):
    if 'random_utility' in game:
        return np.loadtxt('utility/{}/utility{}.csv'.format(game, id), delimiter=',')
    else:
        return np.loadtxt('utility/{}.csv'.format(game), delimiter=',')


def save_utility_matrix(file_name, utility):
    np.savetxt(file_name, utility, fmt='%.8f', delimiter=',')


def save_and_summary_results(dir_name, logs, utilities, xscale, yscale):
    # save log
    for j in range(len(logs)):
        df = logs[j].to_dataframe()
        df.index.name = '#index'
        df.to_csv(dir_name + '/csv/seed{}_results.csv'.format(j))
        save_utility_matrix('{}/csv/seed{}_utility.csv'.format(dir_name, j), utilities[j])

    # calc mean exploitability
    exploitability_dfs = []
    for i in range(len(logs)):
        df = logs[i].to_dataframe()
        exploitability_dfs.append(df['exploitability'])
    exploitability_df = pd.concat(exploitability_dfs, axis=1)

    # save mean exploitability
    df = pd.concat([logs[0].to_dataframe()['iteration'], exploitability_df.mean(axis='columns')], axis=1)
    df.index.name = '#index'
    df.columns = ['iteration', 'exploitability']
    df.to_csv('{}/csv/exploitability_mean.csv'.format(dir_name))

    # plot mean exploitability
    plt.plot(df['iteration'], df['exploitability'], label='Exploitability')
    plt.xlabel('Iterations')
    plt.ylabel('Exploitability')
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(ls='--')
    plt.legend()
    plt.savefig('{}/figure/exploitability_mean.pdf'.format(dir_name))
    plt.close()
