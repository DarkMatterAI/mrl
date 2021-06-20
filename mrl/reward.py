# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/13_reward.ipynb (unless otherwise specified).

__all__ = ['Reward', 'FunctionReward', 'SampleReward', 'NoveltyBonus']

# Cell

from .imports import *
from .torch_imports import *
from .torch_core import *
from .chem import *
from .templates import *
from .agent import *
from .environment import *

# Cell

class Reward(Callback):
    def __init__(self, name, order=10, weight=1., track=True):
        self.name = name
        self.order = order
        self.track = track
        self.weight = weight

    def setup(self):
        log = self.environment.log
        log.add_log(self.name)
        if self.track:
            log.add_metric(self.name)

    def _compute_reward(self):
        raise NotImplementedError

    def compute_reward(self):
        rewards = self._compute_reward()
        rewards = rewards.squeeze()
        rewards = rewards*self.weight
        self.batch_state.rewards += rewards
        self.batch_state[self.name] = rewards

        if self.track:
            self.environment.log.update_metric(self.name, rewards.mean().detach().cpu().numpy())


class FunctionReward(Reward):
    def __init__(self, reward_function, name, order=10, weight=1., track=True):
        super().__init__(name, order, weight, track)
        self.reward_function = reward_function

    def _compute_reward(self):
        return self.reward_function(self.batch_state)


class SampleReward(Reward):
    def __init__(self, reward_function, lookup,
                 name, use_fused=True, order=10, weight=1., track=True):
        super().__init__(name, order, weight, track)
        self.reward_function = reward_function
        self.lookup = lookup
        self.use_fused = use_fused
        self.lookup_table = {}

    def load_data(self, samples, values):
        for i in range(len(samples)):
            self.lookup_table[samples[i]] = values[i]

    def _compute_reward(self):

        if self.use_fused and 'samples_fused' in self.batch_state.keys():
            samples = self.batch_state.samples_fused
        else:
            samples = self.batch_state.samples

        hps = self.batch_state.template_passes
        outputs = to_device(torch.tensor([0. for i in samples]))

        to_score = []
        to_score_idxs = []

        for i, sample in enumerate(samples):
            if self.lookup and sample in self.lookup_table.keys():
                outputs[i] = self.lookup_table[sample]

            else:
                to_score.append(sample)
                to_score_idxs.append(i)

        if to_score:
            scores = self.reward_function(to_score)

            for i in range(len(to_score)):
                outputs[to_score_idxs[i]] = scores[i]

                if self.lookup:
                    item_score = scores[i]
                    if type(item_score) == torch.Tensor:
                        item_score = item_score.detach().cpu()
                    self.lookup_table[to_score[i]] = item_score


        return outputs


# Cell

class NoveltyBonus(Reward):
    def __init__(self, weight, name='novel', order=100, track=True):
        super().__init__(name, order, weight, track)

    def _compute_reward(self):
        log = self.environment.log
        state = self.batch_state
        old = log.unique_samples

        new = [not i in old for i in state.samples]
        reward = to_device(torch.tensor(new)).float()
        return reward
