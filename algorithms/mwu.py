import numpy as np


class MWU(object):
    def __init__(self, eta, n_actions, decay = False):
        self.n_actions = n_actions
        self.logits = np.zeros(n_actions)
        self.strategy = np.ones(n_actions)/n_actions
        self.eta = eta
        self.sum_strategy = np.zeros(n_actions)
        self.decay = decay
        self.update_t = 0

    def update(self, utility):
        if self.decay:
            self.eta = 1/(self.update_t + 1)**(3/4)
        d_strategy = self.eta * utility
        self.logits += d_strategy
        exp_utility = np.exp(d_strategy)*self.strategy
        self.strategy = exp_utility/exp_utility.sum()
        self.update_t += 1


class OMWU(MWU):
    def __init__(self, eta, n_actions, decay = False):
        super().__init__(eta, n_actions, decay)
        self.past_utility = np.zeros(n_actions)

    def update(self, utility):
        if self.decay:
            self.eta = 1/(self.update_t + 1)**(3/4)
        d_strategy = self.eta * (2 * utility - self.past_utility)
        self.logits += d_strategy
        exp_utility = np.exp(d_strategy)*self.strategy
        self.strategy = exp_utility/exp_utility.sum()
        self.update_t += 1
        self.past_utility = utility


class M2WU(MWU):
    def __init__(self, eta, n_actions, mu, update_freq = 0, decay = False):
        super().__init__(eta, n_actions, decay)
        self.mu = mu
        self.reference_strategy = np.ones(n_actions)/n_actions
        self.update_freq = update_freq

    def update(self, utility):
        if self.decay:
            self.eta = 1/(self.update_t + 1)**(3/4)
        d_strategy = self.eta * ( utility +  self.mu * (self.reference_strategy - self.strategy)/self.strategy) 
        exp_utility = np.exp(d_strategy)*self.strategy
        self.strategy = exp_utility/exp_utility.sum()
        # update reference strategy
        if self.update_t > 0 and self.update_freq > 0 and self.update_t % self.update_freq == 0:
            self.reference_strategy = self.strategy.copy()
        self.update_t += 1
