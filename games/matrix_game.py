import numpy as np


class MatrixGame(object):
    def __init__(self, utility, noise_scale=0.1):
        self.utility = utility
        self.noise_scale = noise_scale

    def full_feedback(self, strategies):
        return [self.utility @ strategies[1], -self.utility.T @ strategies[0]]

    def noisy_feedback(self, strategies):
        loss = [self.utility @ strategies[1], -self.utility.T @ strategies[0]]
        loss[0] += np.random.normal(0, self.noise_scale, self.utility.shape[0])
        loss[1] += np.random.normal(0, self.noise_scale, self.utility.shape[1])
        return loss

    def calc_exploitability(self, strategies):
        return max(self.utility @ strategies[1]) + max(-self.utility.T @ strategies[0])

    def num_actions(self, player_id):
        return self.utility.shape[player_id]
