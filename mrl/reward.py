# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/21_reward.ipynb (unless otherwise specified).

__all__ = ['Reward', 'FunctionReward', 'NoveltyReward']

# Cell

from .imports import *
from .core import *
from .callbacks import *
from .torch_imports import *
from .torch_core import *

# Cell

class Reward(Callback):
    def __init__(self, name, sample_name='samples',
                 weight=1., bs=None,
                 order=10, track=True, log=True):
        super().__init__(name=name, order=order)

        self.sample_name = sample_name
        self.weight = weight
        self.bs = bs
        self.track = track
        self.log = log
        self.score_log = {}

    def load_data(self, samples, values):
        for i in range(len(samples)):
            self.score_log[samples[i]] = values[i]

    def setup(self):
        log = self.environment.log
        log.add_log(self.name)
        if self.track:
            log.add_metric(self.name)

    def _compute_reward(self, samples):
        return [0. for i in samples]

    def compute_batched_reward(self, samples):
        if self.bs is not None:
            sample_chunks = chunk_list(samples, self.bs)
            rewards = []
            for chunk in sample_chunks:
                rewards_iter = self._compute_reward(sample_chunks)
                rewards += list(rewards_iter)

        else:
            rewards = self._compute_reward(samples)

        return rewards

    def compute_reward(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state[self.sample_name]

        if self.log:
            to_score = [i for i in samples if not i in self.score_log.keys()]
            rewards = self.compute_batched_reward(to_score)

            for i in range(len(to_score)):
                self.score_log[to_score[i]] = rewards[i]

            rewards = [self.score_log[i] for i in samples]

        else:
            rewards = self.compute_batched_reward(samples)

        rewards = to_device(torch.tensor(rewards).float())
        rewards = rewards * self.weight

        batch_state.rewards += rewards
        batch_state[self.name] = rewards

        if self.track:
            env.log.update_metric(self.name, rewards.mean().detach().cpu().numpy())



# Cell

class FunctionReward(Reward):
    def __init__(self, reward_function, name, sample_name='samples', weight=1.,
                 order=10, track=True, log=True):
        super().__init__(name,
                         sample_name,
                         weight,
                         order,
                         track,
                         log)

        self.reward_function = reward_function


    def _compute_reward(self, samples):
        rewards = []
        if samples:
            rewards = self.reward_function(samples)
        return rewards




# Cell

class NoveltyReward(Reward):
    def __init__(self, weight=1., track=True):
        super().__init__(name='novel', sample_name='samples',
                        weight=weight, order=10, track=track,
                        log=False)

    def _compute_reward(self, samples):
        log = self.environment.log
        old = log.unique_samples
        new = [not i in old for i in samples]
        new = [float(i) for i in new]
        return new