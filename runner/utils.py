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


def save_and_summary_results(dir_name, logs, xscale, yscale):
    # save log
    for j in range(len(logs)):
        df = logs[j].to_dataframe()
        df.index.name = '#index'
        df.to_csv(dir_name + '/csv/seed{}_results.csv'.format(j))

    # calc mean exploitability
    exploitability_dfs = []
    for i in range(len(logs)):
        df = logs[i].to_dataframe()
        exploitability_dfs.append(df['exploitability'])
    df = pd.concat(exploitability_dfs, axis=1).mean(axis='columns')

    # save mean exploitability
    df.index.name = '#index'
    df.to_csv("{}/csv/exploitability_mean.csv".format(dir_name))

    # plot mean exploitability
    plt.plot(df, label='Exploitability')
    plt.xlabel('Iterations')
    plt.ylabel('Exploitability')
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.grid(ls='--')
    plt.legend()
    plt.savefig("{}/figure/exploitability_mean.pdf".format(dir_name))
    plt.close()
