# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/12_environment.ipynb (unless otherwise specified).

__all__ = ['Callback', 'BatchStats', 'Buffer', 'SettrDict', 'BatchState', 'Event', 'Environment', 'Sampler',
           'ModelSampler', 'TemplateCallback', 'AgentCallback', 'GenAgentCallback', 'RewardCallback', 'LossCallback']

# Cell

from .imports import *
from .torch_imports import *
from .torch_core import *
from .chem import *
from .templates import *
from .agent import *

# Cell

class Callback():
    def __init__(self, name='callback', order=1000):
        self.order=order
        self.name = name

    def __call__(self, event_name):

        event = getattr(self, event_name, None)
        if event is not None:
            output = event()
        else:
            output = None

        return output

class BatchStats(Callback):
    def __init__(self):
        super().__init__(name='stats', order=0)

        self.pbar = None
        self.iterations = 0
        self.diversity = []
        self.valid = []
        self.rewards = []
        self.mean_reward = []
        self.metric_vals = ['iterations', 'diversity',
                            'valid', 'rewards', 'mean_reward']
        self.report = 1

    def before_train(self):
        if self.pbar is None:
            print('\t'.join([key for key in self.metric_vals]))
        else:
            self.pbar.write(self.metric_vals, table=True)

    def add_metric(self, name):
        setattr(self, name, [])
        self.metric_vals.append(name)

    def after_batch(self):
        outputs = []
        if self.iterations%self.report==0:
            for metric in self.metric_vals:
                v = getattr(self, metric)
                if type(v)==list:
                    val = v[-1]
                else:
                    val = v

                if type(val)==int:
                    val = f'{val}'
                else:
                    val = f'{val:.2f}'

                outputs.append(val)

            if self.pbar is None:
                print('\t'.join(outputs))
            else:
                self.pbar.write(outputs, table=True)

        self.iterations += 1

class Buffer(Callback):
    def __init__(self, p_total, max_size=1000000):
        super().__init__(name='buffer', order=0)

        self.buffer = []
        self.used_buffer = []
        self.max_size = max_size
        self.p_total = p_total

    def __len__(self):
        return len(self.buffer)

    def add(self, item):

        if is_container(item):
            for i in item:
                self.add(i)
        else:
            place_idx = (len(self.buffer)%self.max_size)+1

            if place_idx>=len(self.buffer):
                self.buffer.append(item)
            else:
                self.buffer[place_idx-1] = item

    def sample(self, n):

        idxs = np.random.choice(np.arange(len(self.buffer)), n, replace=False)
        batch = [self.buffer[i] for i in idxs]
        for idx in sorted(idxs, reverse=True):
            self.buffer.pop(idx)

        self.used_buffer += batch
        self.used_buffer = self.used_buffer[-self.max_size:]

        return batch

    def after_build_buffer(self):
        self.buffer = self.environment.template_cb.filter_sequences(self.buffer)

    def sample_batch(self):
        bs = int(self.environment.bs * self.p_total)
        sample = self.sample(bs)
        self.batch_state.samples += sample
        self.batch_state.sources += ['buffer']*len(sample)

# Cell

class SettrDict(dict):
    def __init__(self):
        super().__init__()

    def __setitem__(self, key, item):
        super().__setitem__(key, item)
        super().__setattr__(key, item)

    def __setattr__(self, key, item):
        super().__setitem__(key, item)
        super().__setattr__(key, item)

    def update_from_dict(self, update_dict):
        for k,v in update_dict.items():
            self[k] = v

class BatchState(SettrDict):
    def __init__(self):
        super().__init__()

        self.samples = []
        self.sources = []
        self.rewards = to_device(torch.tensor(0.))
        self.rewards_scaled = to_device(torch.tensor(0.))
        self.trajectory_rewards = to_device(torch.tensor(0.))
        self.loss = to_device(torch.tensor(0.))
        self.latent_data = []

#         self.sequence_trajectories = []
#         self.x = None
#         self.y = None
#         self.mask = None
#         self.sl = None
#         self.model_output = None
#         self.model_encoded = None
#         self.model_logprobs = None
#         self.model_gathered_logprobs = None
#         self.y_gumbel = None
#         self.vhead_values = None
#         self.old_vhead_values = None
#         self.ref_output = None
#         self.ref_encoded = None
#         self.ref_logprobs = None
#         self.ref_gathered_logprobs = None
#         self.trajectory_rewards = None


# Cell

class Event():
    def __init__(self):
        self.setup = 'setup'
        self.before_train = 'before_train'
        self.build_buffer = 'build_buffer'
        self.after_build_buffer = 'after_build_buffer'
        self.before_batch = 'before_batch'
        self.sample_batch = 'sample_batch'
        self.after_sample = 'after_sample'
        self.get_model_outputs = 'get_model_outputs'
        self.compute_reward = 'compute_reward'
        self.after_compute_reward = 'after_compute_reward'
        self.compute_loss = 'compute_loss'
        self.zero_grad = 'zero_grad'
        self.before_step = 'before_step'
        self.step = 'step'
        self.after_batch = 'after_batch'
        self.after_train = 'after_train'

# Cell

class Environment():
    def __init__(self, agent_cb, template=None, samplers=[], reward_cbs=[], loss_cbs=[], cbs=[],
                buffer_p_batch=None, reward_decay=0.9):
        self.agent_cb = agent_cb
        self.template_cb = TemplateCallback(template)
        self.samplers = samplers
        self.reward_cbs = reward_cbs
        self.loss_cbs = loss_cbs
        self.cbs = []
        if buffer_p_batch is None:
            buffer_p_batch = 1.
            for samp in samplers:
                buffer_p_batch -= samp.p_batch
        self.buffer = Buffer(buffer_p_batch)
        self.batch_state = BatchState()
        self.batch_stats = BatchStats()
        self.mean_reward = None
        self.reward_decay = reward_decay

        all_cbs = [self.agent_cb] + [self.template_cb] + self.samplers + self.reward_cbs
        all_cbs += self.loss_cbs + cbs + [self.buffer] + [self.batch_stats]

        self.register_cbs(all_cbs)
        self('setup')

    def __call__(self, event):
        for cb in self.cbs:
            if hasattr(cb, event):
                cb(event)

    def register_cb(self, cb):
        if isinstance(cb, type):
            cb = cb()
        cb.environment = self
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def register_cbs(self, cbs):
        for cb in cbs:
            self.register_cb(cb)

    def build_buffer(self):
        if len(self.buffer) < self.bs:
            self('build_buffer')
            self('after_build_buffer')

    def sample_batch(self):
        self.batch_state = BatchState()
        for cb in self.cbs:
            cb.batch_state = self.batch_state
        self('before_batch')
        self('sample_batch')
        sequences = self.batch_state.samples
        self.batch_stats.diversity.append(len(set(sequences))/len(sequences))
        self.batch_stats.valid.append(
            len([i for i in sequences if to_mol(i) is not None])/len(sequences))
        self('after_sample')

    def compute_reward(self):
        self('compute_reward')
        rewards = self.batch_state.rewards

        if self.mean_reward is None:
            self.mean_reward = rewards.mean()
        else:
            self.mean_reward = (1-self.reward_decay)*rewards.mean() + self.reward_decay*self.mean_reward

        rewards_scaled = rewards - self.mean_reward
        self.batch_state.rewards_scaled = rewards_scaled
        self.batch_stats.rewards.append(rewards.mean().detach().cpu().numpy())
        self.batch_stats.mean_reward.append(self.mean_reward.detach().cpu().numpy())
        self('after_compute_reward')

    def compute_loss(self):
        self('compute_loss')
        loss = self.batch_state.loss
        self('zero_grad')
        loss.backward()
        self('before_step')
        self('step')

    def fit(self, bs, sl, iters, buffer_size, report):
        self.bs = bs
        self.sl = sl
        self.buffer_size = buffer_size
        self.report = report
        mb = master_bar(range(1))
        self.batch_stats.pbar = mb
        self.batch_stats.report = report
        self('before_train')
        for _ in mb:
            for step in progress_bar(range(iters), parent=mb):
                self.build_buffer()
                self.sample_batch()
                self('get_model_outputs')
                self.compute_reward()
                self.compute_loss()
                self('after_batch')
            self('after_train')


# Cell

class Sampler(Callback):
    def __init__(self, name, p_buffer=0., p_batch=0.):
        super().__init__()
        self.name = name
        self.p_buffer = p_buffer
        self.p_batch = p_batch

    def setup(self):
        if self.p_batch>0.:
            bs = self.environment.batch_stats
            bs.add_metric(f'{self.name}_diversity')
            bs.add_metric(f'{self.name}_valid')
            bs.add_metric(f'{self.name}_rewards')
#             setattr(bs, f'{self.name}_diversity', [])
#             setattr(bs, f'{self.name}_valid', [])
#             setattr(bs, f'{self.name}_rewards', [])
#             setattr(bs, f'{self.name}_new', [])

    def build_buffer(self):
        pass

    def sample_batch(self):
        pass

    def after_compute_reward(self):
        if self.p_batch>0:
            state = self.environment.batch_state
            stats = self.environment.batch_stats
            rewards = state.rewards.detach().cpu().numpy()
            sources = np.array(state.sources)
            if self.name in sources:
                getattr(stats, f'{self.name}_rewards').append(rewards[sources==self.name].mean())
            else:
                getattr(stats, f'{self.name}_rewards').append(0.)


class ModelSampler(Sampler):
    def __init__(self, agent, model, name, p_buffer, p_batch, genbatch, latent=False):
        super().__init__(name, p_buffer, p_batch)
        self.agent = agent
        self.model = model
        self.genbatch = genbatch
        self.latent = latent if self.agent.latents is not None else False

    def build_buffer(self):
        env = self.environment
        bs = int(env.buffer_size * self.p_buffer)
        outputs = []
        to_generate = bs

        if bs > 0:
            for batch in range(int(np.ceil(bs/self.genbatch))):
                current_bs = min(self.genbatch, to_generate)

                preds, _ = self.model.sample_no_grad(current_bs, env.sl, multinomial=True)
                sequences = self.agent.reconstruct(preds)
                sequences = list(set(sequences))
                sequences = [i for i in sequences if to_mol(i) is not None]
                outputs += sequences
                outputs = list(set(outputs))
                to_generate = bs - len(outputs)

            env.buffer.add(outputs)


    def sample_batch(self):
        env = self.environment
        bs = int(env.bs * self.p_batch)

        if bs > 0:

            if self.latent:
                latents = self.agent.latents
                latent_idxs = torch.randint(0, latents.shape[0]-1, bs)
                sample_latents = latents[latent_idxs]
                self.batch_state.latent_data.append([self.name, latent_idxs])
            else:
                sample_latents=None


            preds, _ = self.model.sample_no_grad(bs, env.sl, z=sample_latents, multinomial=True)
            sequences = self.agent.reconstruct(preds)
            diversity = len(set(sequences))/len(sequences)
            valid = np.array([to_mol(i) is not None for i in sequences])
            getattr(env.batch_stats, f"{self.name}_diversity").append(diversity)
            getattr(env.batch_stats, f"{self.name}_valid").append(valid.mean())
            self.batch_state.samples += sequences
            self.batch_state.sources += [self.name]*len(sequences)


# Cell

class TemplateCallback(Callback):
    def __init__(self, template=None):
        super().__init__(order=-1)
        self.template = template
        self.name = 'template'

    def compute_reward(self):
        env = self.environment
        state = env.batch_state

        if self.template is not None:
            rewards = np.array(self.template.eval_mols(state.samples))
            hps = np.array(self.template(state.samples))
        else:
            rewards = np.array([0.]*len(state.samples))
            hps = np.array([0.]*len(state.samples))

        state.template_rewards = rewards
        state.template_passes = hps
        state.rewards += to_device(torch.from_numpy(rewards).float())

    def filter_sequences(self, sequences):
        if self.template is not None:
            hp = np.array(self.template(sequences))
            sequences = np.array(sequences)[hp]
            sequences = list(sequences)

        return sequences

# Cell

class AgentCallback(Callback):
    def __init__(self, agent, name):
        super().__init__()
        self.agent = agent
        self.name = name

    def zero_grad(self):
        self.agent.zero_grad()

    def before_step(self):
        nn.utils.clip_grad_norm_(self.agent.model.parameters(), 1.)

    def step(self):
        self.agent.step()

    def after_sample(self):
        # convert samples to tensors
        raise NotImplementedError

    def get_model_outputs(self):
        # get relevant model outputs
        raise NotImplementedError

class GenAgentCallback(AgentCallback):
    def __init__(self, agent, name, contrastive=False):
        super().__init__(agent, name)
        self.contrastive = contrastive

    def after_sample(self):

        batch_ds = self.agent.dataset.new(self.batch_state.samples)
        batch = batch_ds.collate_function([batch_ds[i] for i in range(len(batch_ds))])
        x,y = batch

        self.batch_state.x = x
        self.batch_state.y = y
        mask = ~(y==self.agent.vocab.stoi['pad'])
        self.batch_state.mask = mask
        self.batch_state.lengths = mask.sum(-1)
        self.batch_state.sl = y.shape[-1]
        self.batch_state.sequence_trajectories = self.agent.reconstruct_trajectory(y)
        self.batch_state.rewards = to_device(torch.zeros(x.shape[0]))
        self.batch_state.rewards_scaled = to_device(torch.zeros(x.shape[0]))
        self.batch_state.trajectory_rewards = to_device(torch.zeros(y.shape))

    def get_model_outputs(self):

        x = self.batch_state.x
        y = self.batch_state.y
        sources = self.batch_state.sources
        latent_info = self.batch_state.latent_data

        if latent_info:
            latent_sources = []
            output_tensors = []
            for (latent_source, latent_idxs) in latent_info:
                latent_sources.append(latent_source)
                latent_mask = torch.tensor([i==latent_source for i in sources]).bool()
                latents = self.agent.latents[latent_idxs]
                out = self.agent.model.get_rl_tensors(x[latent_mask], y[latent_mask],
                                                      latents=latents)
                output_tensors.append(out)

            non_latent_mask = torch.tensor([not i in latent_sources for i in sources]).bool()
            out = self.agent.model.get_rl_tensors(x[non_latent_mask], y[non_latent_mask])
            output_tensors.append(out)

            mo = torch.cat([i[0] for i in output_tensors], 0)
            mlp = torch.cat([i[1] for i in output_tensors], 0)
            mglp = torch.cat([i[2] for i in output_tensors], 0)
            me = torch.cat([i[3] for i in output_tensors], 0)

        else:
            mo, mlp, mglp, me = self.agent.model.get_rl_tensors(x,y)

        mprob = mlp.exp()

        self.batch_state.model_output = mo
        self.batch_state.model_logprobs = mlp
        self.batch_state.model_gathered_logprobs = mglp
        self.batch_state.model_encoded = me
        self.batch_state.y_gumbel = F.one_hot(y, len(self.agent.vocab.itos)) + mprob - mprob.detach()

        if self.agent.value_head is not None:
            value_predictions = self.agent.value_head(me)
            with torch.no_grad():
                base_value_predictions = self.agent.base_value_head(me)
        else:
            value_predictions = None
            base_value_predictions = None

        self.batch_state.state_values = value_predictions
        self.batch_state.ref_state_values = base_value_predictions

        if self.agent.base_model is not None:
            with torch.no_grad():
                bo, blp, bglp, be = self.agent.base_model.get_rl_tensors(x,y)
        else:
            bo, blp, bglp, be = None, None, None, None

        self.batch_state.reference_output = bo
        self.batch_state.reference_logprobs = blp
        self.batch_state.reference_gathered_logprobs = bglp
        self.batch_state.reference_encoded = be



# Cell

class RewardCallback(Callback):
    def __init__(self, reward_function, name, weight=1.):
        super().__init__(order=1)
        self.name = name
        self.reward_function = reward_function
        self.weight = weight

    def setup(self):
        bs = self.environment.batch_stats
        bs.add_metric(self.name)
#         setattr(bs, self.name, [])

    def compute_reward(self):
        rewards, reward_dict = self.reward_function.from_batch_state(self.batch_state)
        getattr(self.environment.batch_stats, self.name).append(rewards.mean().detach().cpu().numpy())
        rewards = rewards * self.weight
        self.batch_state.rewards += rewards
        self.batch_state[self.name] = reward_dict

class LossCallback(Callback):
    def __init__(self, loss_function, name, weight=1.):
        super().__init__(order=1)
        self.name = name
        self.loss_function = loss_function
        self.weight = weight

    def setup(self):
        bs = self.environment.batch_stats
        bs.add_metric(self.name)
#         setattr(bs, self.name, [])

    def compute_loss(self):
        loss, loss_dict = self.loss_function.from_batch_state(self.batch_state)
        getattr(self.environment.batch_stats, self.name).append(loss.detach().cpu().numpy())
        loss = loss * self.weight
        self.batch_state.loss += loss
        self.batch_state[self.name] = loss_dict