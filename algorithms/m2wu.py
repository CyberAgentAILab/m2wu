import numpy as np

from algorithms import MWU


class M2WU(MWU):
    def __init__(self, n_actions, random_init_strategy, eta, mu, update_freq=0, decay=False):
        super().__init__(n_actions, random_init_strategy, eta, decay)
        self.mu = mu
        self.ref_strategy = np.ones(n_actions) / n_actions
        self.update_freq = update_freq

    def update(self, utility):
        if self.decay:
            self.eta = 1 / (self.t + 1) ** (3 / 4)
        values = np.exp(self.eta * (utility + self.mu * (self.ref_strategy - self.strategy) / self.strategy)) * self.strategy
        self.strategy = values / values.sum()
        self.t += 1
        self._update_ref_strategy()

    def _update_ref_strategy(self):
        if self.update_freq is not None and self.t % self.update_freq == 0:
            self.ref_strategy = self.strategy.copy()

    @classmethod
    def name(cls, params):
        alg_name = cls.__name__
        if params['update_freq'] is not None:
            alg_name += '_uf{}'.format(params['update_freq'])
        return alg_name
