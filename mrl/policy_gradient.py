# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/11_policy_gradients.ipynb (unless otherwise specified).

__all__ = ['BasePolicy', 'PolicyGradient', 'TRPO', 'PPO']

# Cell

from .imports import *
from .torch_imports import *
from .torch_core import *
from .layers import *

# Cell

class BasePolicy():
    def __init__(self, gamma=1.):
        self.gamma = gamma

    def discount_rewards(self, model_outputs):
        rewards = model_outputs['rewards_scaled']
        mask = model_outputs['mask']
        rewards = scatter_rewards(rewards, mask)

        traj_rewards = model_outputs['trajectory_rewards']
        if traj_rewards is not None:
            rewards += traj_rewards

        discounted = discount_rewards(rewards, self.gamma)

        return discounted

# Cell

class PolicyGradient(BasePolicy):
    def __init__(self, discount=True, gamma=0.97):
        super().__init__(gamma)
        self.discount = discount

    def __call__(self, model_outputs):

        lps = model_outputs['model_gathered_logprobs']
        mask = model_outputs['mask']
        rewards = model_outputs['rewards_scaled']

        if not self.discount:
            pg_loss = -((lps*mask).sum(-1).sum(-1)*rewards)/mask.sum(-1)
        else:
            rewards = self.discount_rewards(model_outputs)
            rewards = whiten(rewards)
            pg_loss = -(lps*rewards*mask).sum(-1)/mask.sum(-1)

        model_outputs['pg_loss'] = pg_loss.mean()
        model_outputs['pg_dict'] = {'pg_rewards' : rewards}

        return model_outputs

# Cell

class TRPO(BasePolicy):
    def __init__(self, gamma, kl_target, beta=1., eta=50, lam=0.95, v_coef=0.5):
        self.gamma = gamma
        self.beta = beta
        self.eta = eta
        self.lam = lam
        self.kl_target = kl_target
        self.v_coef = v_coef

    def __call__(self, model_outputs):
        discounted_rewards = self.discount_rewards(model_outputs)

        values = model_outputs['state_values']
        advantages = self.compute_advantages(discounted_rewards, values)
        advantages = whiten(advantages)

        v_loss = self.value_loss(values, discounted_rewards)

        lps = model_outputs['model_gathered_logprobs']
        ref_lps = model_outputs['reference_gathered_logprobs']
        mask = model_outputs['mask']

        ratios = (lps - ref_lps).exp()

        loss1 = -(ratios*advantages*mask).sum(-1)/mask.sum(-1)

        kl = torch.distributions.kl.kl_divergence(
                    Categorical(logits=model_outputs['reference_logprobs']),
                    Categorical(logits=model_outputs['model_logprobs']))

        kl = (kl*mask).sum(-1)/mask.sum(-1)
        kl = kl.mean()

        loss2 = self.beta*kl

        loss3 = self.eta * torch.maximum(torch.tensor(0.), kl - 2.0*self.kl_target)

        loss1 = loss1.mean()
        loss3 = loss3.mean()

        pg_loss = loss1 + loss2 + loss3 + v_loss

        model_outputs['pg_loss'] = pg_loss
        model_outputs['pg_dict'] = {'pg_discounted' : discounted_rewards,
                                    'pg_advantage' : advantages,
                                    'ratios' : ratios.detach().cpu(),
                                    'kl' : kl.detach().cpu(),
                                    'loss1' : loss1.detach().cpu(),
                                    'loss2' : loss2.detach().cpu(),
                                    'loss3' : loss3.detach().cpu(),
                                    'v_loss' : v_loss.detach().cpu()}

        return model_outputs

    def compute_advantages(self, rewards, values):

        if values is None:
            advantages = rewards
        else:
            advantages = compute_advantages(rewards, values.detach(), self.gamma, self.lam)

        return advantages

    def value_loss(self, values, rewards):
        if values is None:
            v_loss = torch.tensor(0.)
        else:
            v_loss = self.v_coef*F.mse_loss(values, rewards)

        return v_loss

# Cell

class PPO(BasePolicy):
    def __init__(self, gamma, kl_coef, lam=0.95, v_coef=0.5, cliprange=0.2, ent_coef=0.01,
                 kl_target=None, kl_horizon=None):
        self.gamma = gamma
        self.lam = lam
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.kl_horizon = kl_horizon
        self.v_coef = v_coef
        self.cliprange = cliprange

    def __call__(self, model_outputs):
        discounted_rewards = self.discount_rewards(model_outputs)

        kl_reward = self.compute_kl_reward(model_outputs)
        discounted_rewards = discounted_rewards + kl_reward

        values = model_outputs['state_values']
        advantages = self.compute_advantages(discounted_rewards, values)
        advantages = whiten(advantages)

        values = model_outputs['state_values']
        v_loss = self.value_loss(values, discounted_rewards)

        lps = model_outputs['model_gathered_logprobs']
        ref_lps = model_outputs['reference_gathered_logprobs']
        mask = model_outputs['mask']

        ratios = (lps - ref_lps).exp()
        ratios_clipped = torch.clamp(ratios, 1.0-self.cliprange, 1.0+self.cliprange)

        loss1 = -(ratios*advantages)
        loss2 = -(ratios_clipped*advantages)

        loss = torch.maximum(loss1, loss2)
        loss = (loss*mask).sum(-1)/mask.sum(-1)

        entropy = Categorical(lps).entropy().mean()

        loss = loss.mean()

        pg_loss = loss + v_loss - self.ent_coef*entropy

        self.update_kl(model_outputs)

        model_outputs['pg_loss'] = pg_loss
        model_outputs['pg_dict'] = {'pg_discounted' : discounted_rewards,
                                    'pg_advantage' : advantages,
                                    'ratios' : ratios.detach().cpu(),
                                    'loss' : loss.detach().cpu(),
                                    'v_loss' : v_loss.detach().cpu(),
                                    'entropy' : entropy.detach().cpu()}

        return model_outputs

    def compute_kl_reward(self, model_outputs):
        lps = model_outputs['model_gathered_logprobs']
        ref_lps = model_outputs['reference_gathered_logprobs']
        kl = lps - ref_lps
        kl_reward = -self.kl_coef * kl.detach()
        return kl_reward

    def value_loss(self, values, rewards):
        if values is None:
            v_loss = torch.tensor(0.)
        else:
            v_loss = self.v_coef*F.mse_loss(values, rewards)

        return v_loss

    def compute_advantages(self, rewards, values):

        if values is None:
            advantages = rewards
        else:
            advantages = compute_advantages(rewards, values.detach(), self.gamma, self.lam)

        return advantages

    def update_kl(self, model_outputs):

        if (self.kl_target is not None) and (self.kl_horizon is not None):
            lps = model_outputs['model_gathered_logprobs']
            ref_lps = model_outputs['reference_gathered_logprobs']
            mask = model_outputs['mask']
            kl = (lps - ref_lps).detach()
            kl = (kl*mask).sum(-1)/mask.sum(-1)

            error = np.clip(kl/self.kl_target - 1, -0.2, 0.2)
            factor = 1 + errror * lps.shape[0]/self.kl_horizon
            self.kl_coef *= factor
