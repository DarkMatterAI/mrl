# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/18_losses.ipynb (unless otherwise specified).

__all__ = ['LossCallback', 'PolicyLoss', 'PriorLoss', 'HistoricPriorLoss']

# Cell

from .imports import *
from .core import *
from .torch_imports import *
from .torch_core import *
from .callbacks import *
from .policy_gradient import *

# Cell

class LossCallback(Callback):
    def __init__(self, loss_function, name, weight=1., track=True, order=20):
        super().__init__(name=name, order=order)
        self.loss_function = loss_function
        self.weight = weight
        self.track = track

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)
            log.add_log(self.name)

    def compute_loss(self):
        loss = self.loss_function.from_batch_state(self.environment.batch_state)
        loss = loss * self.weight

        if self.track:
            self.environment.log.update_metric(self.name, loss.mean().detach().cpu().numpy())

        self.environment.batch_state.loss += loss.mean()
        self.environment.batch_state[self.name] = loss.detach().cpu().numpy()

# Cell

class PolicyLoss(LossCallback):
    def __init__(self, policy_function, name, value_head=None,
                 v_update=0.95, v_update_iter=10,
                 vopt_kwargs={}, weight=1, track=True):
        assert isinstance(policy_function, BasePolicy)
        super().__init__(policy_function, name, weight, track)

        self.set_model(value_head, vopt_kwargs)
        self.v_update = v_update
        self.v_update_iter = v_update_iter
        self.fields = [
            'model_gathered_logprobs',
            'base_gathered_logprobs',
            'mask',
            'trajectory_rewards',
            'model_logprobs',
            'base_logprobs',
            'value_input'
        ]

    def set_model(self, value_head, vopt_kwargs):
        self.value_head = value_head
        if self.value_head is not None:
            self.base_value_head = copy.deepcopy(self.value_head)
            to_device(self.value_head)
            to_device(self.base_value_head)
            self.opt = optim.Adam(self.value_head.parameters(), **vopt_kwargs)
        else:
            self.opt = None

    def after_sample(self):
        env = self.environment
        batch_state = env.batch_state
        for field in self.fields:
            if not hasattr(batch_state, field):
                batch_state[field] = None

    def get_model_outputs(self):

        env = self.environment
        batch_state = env.batch_state
        value_input = batch_state.value_input

        if (self.value_head is not None) and (value_input is not None):
            value_predictions = self.value_head(value_input)
            with torch.no_grad():
                base_value_predictions = self.base_value_head(value_input)

        else:
            value_predictions = None
            base_value_predictions = None

        batch_state.state_values = value_predictions
        batch_state.ref_state_values = base_value_predictions

    def zero_grad(self):
        if self.opt is not None:
            self.opt.zero_grad()

    def step(self):
        if self.opt is not None:
            self.opt.step()

    def after_batch(self):
        log = self.environment.log
        iterations = log.iterations
        if iterations%self.v_update_iter == 0 and iterations>0:
            self.update_base_model()

    def update_base_model(self):
        if self.value_head is not None:
            if self.v_update < 1:
                merge_models(self.base_value_head, self.value_head, alpha=self.v_update)

    def save_weights(self, filename):
        state_dict = {}

        if isinstance(self.value_head, nn.Module):
            state_dict['value_head'] = self.value_head.state_dict()
            state_dict['base_value_head'] = self.base_value_head.state_dict()
        else:
            state_dict['value_head'] = None
            state_dict['base_value_head'] = None

        torch.save(state_dict, filename)

    def load_weights(self, filename):
        state_dict = torch.load(filename, map_location=get_model_device(self.model))

        if isinstance(self.value_head, nn.Module):
            self.value_head.load_state_dict(state_dict['value_head'])
            self.base_value_head.load_state_dict(state_dict['base_value_head'])


# Cell

class PriorLoss():
    def __init__(self, prior, base_prior=None, clip=10.):

        self.prior = prior
        self.base_prior = base_prior
        self.clip = clip

    def loss(self, z, rewards):
        rewards = rewards-rewards.mean()

        prior_lps = self.prior.log_prob(z)

        if self.base_prior is not None:
            with torch.no_grad():
                base_lps = self.base_prior.log_prob(z.detach())

            ratios = prior_lps - base_lps.detach()
            ratios = torch.clip(ratios, -self.clip, self.clip)

            prior_loss = (-ratios.mean(-1)*rewards)
        else:
            prior_lps = torch.clip(prior_lps, -self.clip, self.clip)
            prior_loss = (-prior_lps.mean(-1)*rewards)

#         prior_loss = torch.clip(prior_loss, -self.clip, self.clip)

        return prior_loss

    def from_batch_state(self, batch_state, subset_name=None):
        z = batch_state.model_latent
        rewards = batch_state.rewards

        if subset_name is not None:
            sources = batch_state.sources
            sources = np.array([i.replace('_buffer', '') for i in sources])
            source_mask = sources==subset_name

            loss = to_device(torch.zeros(sources.shape))

            z = z[source_mask]
            rewards = rewards[source_mask]

            if z.numel()>0:
                loss[source_mask] = self.loss(z, rewards)

        else:
            loss = self.loss(z, rewards)

        return loss


class HistoricPriorLoss(Callback):
    def __init__(self, prior_loss, model, dataset, percentile,
                 n, above_percent, start_iter, frequency,
                 log_term='rewards', weight=1., track=True):
        super().__init__(name='hist_prior', order=20)

        if not is_container(prior_loss):
            prior_loss = [prior_loss]

        self.prior_loss = prior_loss
        self.model = model
        self.dataset = dataset
        self.percentile = percentile
        self.n = n
        self.above_percent = above_percent
        self.start_iter = start_iter
        self.frequency = frequency
        self.log_term = log_term
        self.weight = weight
        self.track = track

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)

    def compute_loss(self):
        loss = self.historic_loss()
        loss = loss * self.weight

        if self.track:
            self.environment.log.update_metric(self.name, loss.mean().detach().cpu().numpy())

        self.environment.batch_state.loss += loss.mean()
        self.environment.batch_state[self.name] = loss.detach().cpu().numpy()

    def select_data(self):
        env = self.environment
        df = env.log.df

        df1 = df[df[self.log_term]>np.percentile(df[self.log_term].values, self.percentile)]
        n_samp = min(int(self.n*self.above_percent), df1.shape[0])
        samples1 = df1.sample(n=n_samp)

        df2 = df[df[self.log_term]<np.percentile(df[self.log_term].values, self.percentile)]
        n_samp = min(int(self.n*(1-self.above_percent)), df2.shape[0])
        samples2 = df2.sample(n=n_samp)

        df = pd.concat([samples1, samples2])
        return df

    def historic_loss(self):
        env = self.environment

        iterations = self.environment.log.iterations

        if (iterations > self.start_iter) and (iterations%self.frequency==0):

            df = self.select_data()

            rewards = to_device(torch.tensor(df.rewards.values).float())

            batch_ds = self.dataset.new(df.samples.values)
            batch = batch_ds.collate_function([batch_ds[i] for i in range(len(batch_ds))])
            batch = to_device(batch)
            x,y = batch

            with torch.no_grad():
                z = self.model.x_to_latent(x)

            prior_loss = sum([i.loss(z, rewards) for i in self.prior_loss])

        else:
            prior_loss = torch.tensor(0.)

        return prior_loss