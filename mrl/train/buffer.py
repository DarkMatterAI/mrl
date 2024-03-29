# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/14_buffer.ipynb (unless otherwise specified).

__all__ = ['Buffer', 'WeightedBuffer', 'PredictiveBuffer', 'BufferSizeCallback']

# Cell

from ..imports import *
from ..core import *
from .callback import *
from ..torch_imports import *
from ..torch_core import *

# Cell

class Buffer(Callback):
    '''
    Buffer - training buffer

    Inputs:

    - `p_total float`: batch percentage for `sample_batch`
    '''
    def __init__(self, p_total):
        super().__init__(name='buffer', order=0)

        self.buffer = []
        self.buffer_sources = []
        self.p_total = p_total

    def __len__(self):
        return len(self.buffer)

    def add(self, item, name=''):

        if type(item)==list:
            for i in item:
                self.add(i, name=name)
        else:
            self.buffer.append(item)
            self.buffer_sources.append(name+'_buffer')

    def sample(self, n):

        idxs = np.random.choice(np.arange(len(self.buffer)), min(n, len(self.buffer)),
                                replace=False)
        batch = [self.buffer[i] for i in idxs]
        sources = [self.buffer_sources[i] for i in idxs]
        for idx in sorted(idxs, reverse=True):
            self.buffer.pop(idx)
            self.buffer_sources.pop(idx)

        return batch, sources

    def _filter_buffer(self, valids):

        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if valids[i]]
        self.buffer_sources = [self.buffer_sources[i]
                               for i in range(len(self.buffer_sources)) if valids[i]]

    def filter_buffer(self):
        if self.buffer:
            seen = set()
            unique = []
            for item in self.buffer:
                if item in seen:
                    unique.append(False)
                else:
                    seen.add(item)
                    unique.append(True)

            self._filter_buffer(np.array(unique))

    def sample_batch(self):
        env = self.environment
        batch_state = env.batch_state

        bs = int(env.bs * self.p_total)
        if bs>0:
            sample, sources = self.sample(bs)
            batch_state.samples += sample
            batch_state.sources += sources
        elif bs==-1:
            sample = self.buffer
            sources = self.buffer_sources
            batch_state.samples += sample
            batch_state.sources += sources
            self.buffer = []
            self.buffer_sources = []

    def filter_batch(self):
        env = self.environment
        batch_state = env.batch_state
        samples = batch_state.samples

        unique_samples = set()
        unique = []

        for sample in samples:
            if sample in unique_samples:
                unique.append(False)
            else:
                unique_samples.add(sample)
                unique.append(True)

        unique = np.array(unique)
        self._filter_batch(unique)


# Cell

class WeightedBuffer(Buffer):
    '''
    WeightedBuffer - base class for buffer with
    weighted sampling

    Inputs:

    - `p_total float`: batch percentage for `sample_batch`

    - `refresh_predictions int`: how often to generate
    new prdictions for all items in the buffer

    - `pct_argmax float[0., 1.]`: percent of samples to draw
    with argmax over the calculated weight versus weighted
    random sampling
    '''
    def __init__(self, p_total, refresh_predictions, pct_argmax=0.):
        super().__init__(p_total)

        self.weights = []
        self.refresh_predictions = refresh_predictions
        self.pct_argmax = pct_argmax
        self.name = 'weighted_buffer'

    def add(self, item, name=''):

        if type(item)==list:
            for i in item:
                self.add(i, name=name)
        else:
            self.buffer.append(item)
            self.buffer_sources.append(name+'_buffer')
            self.weights.append(None)

    def compute_weights(self, samples):
        raise NotImplementedError

    def _filter_buffer(self, valids):

        self.buffer = [self.buffer[i] for i in range(len(self.buffer)) if valids[i]]
        self.buffer_sources = [self.buffer_sources[i]
                               for i in range(len(self.buffer_sources)) if valids[i]]
        self.weights = [self.weights[i] for i in range(len(self.weights)) if valids[i]]

    def sample(self, n):
        weights = np.array(self.weights)

        all_idxs = np.arange(len(self.buffer))
        idxs = []

        if self.pct_argmax>0.:
            n_argmax = int(n*self.pct_argmax)
            idxs_sorted = np.argsort(weights)
            argmax_idxs = idxs_sorted[-n_argmax:]
            idxs += list(argmax_idxs)
            n = n - n_argmax
            all_idxs = idxs_sorted[:-n_argmax]
            weights = weights[all_idxs]

        if weights.shape[0]>0:
            weights = weights - weights.min() + 1e-8
            weights = weights / weights.sum()

            sampled_idxs = np.random.choice(all_idxs, min(n, len(all_idxs)),
                                            replace=False, p=weights)
            idxs += list(sampled_idxs)

        batch = [self.buffer[i] for i in idxs]
        sources = [self.buffer_sources[i] for i in idxs]
        weights = [self.weights[i] for i in idxs]
        for idx in sorted(idxs, reverse=True):
            self.buffer.pop(idx)
            self.buffer_sources.pop(idx)
            self.weights.pop(idx)

        return batch, sources, weights

    def before_batch(self):

        weights = np.array(self.weights)

        idxs = np.arange(weights.shape[0])
        to_score = weights==None
        to_score_idxs = idxs[to_score]
        to_score_samples = [self.buffer[i] for i in to_score_idxs]

        if to_score_samples:
            scored_weights = self.compute_weights(to_score_samples)
            weights[to_score_idxs] = scored_weights

        weights = list(weights)
        self.weights = weights

    def sample_batch(self):
        env = self.environment
        batch_state = env.batch_state

        bs = int(env.bs * self.p_total)
        if bs>0:
            sample, sources, weights = self.sample(bs)
            batch_state.samples += sample
            batch_state.sources += sources
            batch_state.buffer_weights = weights
        elif bs==-1:
            sample = self.buffer
            sources = self.buffer_sources
            weights = self.weights
            batch_state.samples += sample
            batch_state.sources += sources
            batch_state.buffer_weights = weights
            self.buffer = []
            self.buffer_sources = []
            self.weights = []

    def after_batch(self):
        env = self.environment
        iterations = env.log.iterations

        if iterations>0 and iterations%self.refresh_predictions==0:
            weights = self.compute_weights(self.buffer)

            self.weights = list(weights)

# Cell

class PredictiveBuffer(WeightedBuffer):
    '''
    PredictiveBuffer - buffer with active learning
    score prediction

    Inputs:

    - `p_total float`: batch percentage for `sample_batch`

    - `refresh_predictions int`: how often to generate
    new prdictions for all items in the buffer

    - `predictive_agent PredictiveAgent`: active learning
    agent to train

    - `pred_bs int`: prediction batch size for `predictive_agent`

    - `supervised_frequency int`: how often to run
    offline supervised training of the predictive agent

    - `supervised_epochs int`: how many epochs to run
    during offline supervised training

    - `supervised_bs int`: batch size for offline
    supervised training

    - `supervised_lr float`: learning rate for
    offline supervised training

    - `train_silent bool`: if True, offline supervised training
    results are printed

    - `pct_argmax float[0., 1.]`: percent of samples to draw
    with argmax over the calculated weight versus weighted
    random sampling

    - `track bool`: if True, predictive buffer metrics
    are added to the environment printout
    '''
    def __init__(self, p_total, refresh_predictions, predictive_agent, pred_bs,
                 supervised_frequency, supervised_epochs,
                 supervised_bs, supervised_lr, train_silent=True,
                 pct_argmax=0., track=True):
        super().__init__(p_total=p_total,
                         refresh_predictions=refresh_predictions,
                         pct_argmax=pct_argmax)

        self.predictive_agent = predictive_agent
        unfreeze(self.predictive_agent.model)
        self.pred_bs = pred_bs

        self.supervised_frequency = supervised_frequency
        self.supervised_epochs = supervised_epochs
        self.supervised_bs = supervised_bs
        self.supervised_lr = supervised_lr
        self.train_silent = train_silent

        self.track = track
        self.name = 'predictive_buffer'

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name+'_loss')
            log.add_metric(self.name+'_preds')
            log.add_log(self.name+'_preds')

    def compute_weights(self, samples):
        with torch.no_grad():
            if len(samples) < self.pred_bs:
                scores = self.predictive_agent.predict_data(samples, detach=True).squeeze()
                scores = scores.detach().cpu().numpy()
            else:
                scores = self.predictive_agent.predict_data_batch(samples, self.pred_bs, detach=True).squeeze()
                scores = scores.detach().cpu().numpy()
        return scores

    def get_model_outputs(self):
        env = self.environment
        samples = env.batch_state.samples
        preds = self.predictive_agent.predict_data(samples).squeeze()
        env.batch_state[self.name+'_preds'] = preds

        if self.track:
            env.log.update_metric(self.name+'_preds', preds.mean().detach().cpu().numpy())

    def compute_loss(self):
        env = self.environment
        rewards = env.batch_state.rewards
        preds = env.batch_state[self.name+'_preds']
#         preds = preds.to(rewards.device)
        loss = self.predictive_agent.loss_function(preds, rewards)

        if self.track:
            env.log.update_metric(self.name+'_loss', loss.mean().detach().cpu().numpy())

        env.batch_state.loss += loss.mean()

    def zero_grad(self):
        self.predictive_agent.zero_grad()

    def before_step(self):
        self.predictive_agent.before_step()

    def step(self):
        self.predictive_agent.step()

    def after_batch(self):
        env = self.environment
        iterations = self.environment.log.iterations

        if iterations>0 and iterations%self.supervised_frequency==0 and self.supervised_frequency>0:
            self.train_model()

        if iterations>0 and iterations%self.refresh_predictions==0:
            weights = self.compute_weights(self.buffer)

            self.weights = list(weights)

    def train_model(self):
        env = self.environment
        df = env.log.df[['samples', 'rewards']]
        self.predictive_agent.update_dataset_from_inputs(df.samples.values, df.rewards.values)
        self.predictive_agent.train_supervised(self.supervised_bs, self.supervised_epochs,
                                              self.supervised_lr, silent=self.train_silent)


# Cell

class BufferSizeCallback(Callback):
    '''
    BufferSizeCallback - print out
    current buffer size during training
    '''
    def __init__(self):
        super().__init__(name='buffer size')

    def setup(self):
        log = self.environment.log
        log.add_metric(self.name)

    def after_sample(self):
        env = self.environment
        buffer_size = len(env.buffer)
        self.environment.log.update_metric(self.name, buffer_size)