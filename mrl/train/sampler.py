# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/19_samplers.ipynb (unless otherwise specified).

__all__ = ['Sampler', 'DatasetSampler', 'ModelSampler', 'PriorSampler', 'LatentSampler', 'ContrastiveSampler',
           'LogSampler', 'TokenSwapSampler', 'LogEnumerator', 'Timeout', 'MurckoTimeout']

# Cell

from ..imports import *
from ..torch_imports import *
from ..torch_core import *
from .callback import *
from ..chem import *

# Cell

class Sampler(Callback):
    '''
    Sampler - base sampler callback

    Inputs:

    - `name str`: sampler name

    - `buffer_size int`: how many samples to add
    during `build_buffer`

    - `p_batch float`: what percentage of batch
    samples should come from this sampler

    - `track bool`: if metrics from this sampler
    should be tracked
    '''
    def __init__(self, name, buffer_size=0, p_batch=0., track=True):
        super().__init__(name=name)
        self.name = name
        self.buffer_size = buffer_size
        self.p_batch = p_batch
        self.track = track

    def _build_buffer(self):
        return []

    def _sample_batch(self):
        return []

    def build_buffer(self):
        outputs = self._build_buffer()
        if outputs:
            self.environment.buffer.add(outputs, self.name)

    def sample_batch(self):
        outputs = self._sample_batch()
        if outputs:
            self.environment.batch_state.samples += outputs
            self.environment.batch_state.sources += [self.name]*len(outputs)


# Cell

class DatasetSampler(Sampler):
    '''
    DatasetSampler - adds items from `data`
    to either the buffer or the current batch

    Inputs:

    - `data list`: list of data points to sample from

    - `name str`: sampler name

    - `buffer_size int`: how many samples to add
    during `build_buffer`

    - `p_batch float`: what percentage of batch
    samples should come from this sampler
    '''
    def __init__(self, data, name, buffer_size=0, p_batch=0.):
        super().__init__(name, buffer_size, p_batch)
        self.data = data

    def sample_data(self, n):
        n = min(n, len(self.data))
        idxs = np.random.randint(0, len(self.data), n)
        samples = [self.data[i] for i in idxs]
        return samples

    def _build_buffer(self):
        samples = []
        if self.buffer_size>0:
            samples = self.sample_data(self.buffer_size)
        return samples

    def _sample_batch(self):
        samples = []
        bs = self.environment.bs
        bs_sample = int(self.p_batch*bs)

        if bs_sample > 0:
            sample = self.sample_data(bs_sample)

        return samples

# Cell

class ModelSampler(Sampler):
    '''
    ModelSampler - sampler class to draw samples from a `GenerativeModel`

    Inputs:

    - `vocab Vocab`: vocabulary for reconstructing samples

    - `model GenerativeModel`: model to sample from

    - `name str`: sampler name

    - `buffer_size int`: number of samples to generate during `build_buffer`

    - `p_batch float`: what percentage of batch
    samples should come from this sampler

    - `genbatch int`: generation batch size

    - `track bool`: if metrics from this sampler should be tracked

    - `temperature float`: sampeling temperature
    '''
    def __init__(self, vocab, model, name, buffer_size, p_batch,
                 genbatch, track=True, temperature=1.):
        super().__init__(name, buffer_size, p_batch, track)

        self.vocab = vocab
        self.model = model
        self.genbatch = genbatch
        self.temperature = temperature

    def setup(self):
        if self.p_batch>0. and self.track:
            log = self.environment.log
            log.add_metric(f'{self.name}_diversity')
            log.add_metric(f'{self.name}_valid')
            log.add_metric(f'{self.name}_rewards')
            log.add_metric(f'{self.name}_new')

    def build_buffer(self):
        env = self.environment
        sl = env.sl
        outputs = self._build_buffer(sl)
        if outputs:
            self.environment.buffer.add(outputs, self.name)

    def _build_buffer(self, sl):
        buffer_size = self.buffer_size
        outputs = []
        to_generate = buffer_size

        if buffer_size > 0:
            for batch in range(int(np.ceil(buffer_size/self.genbatch))):
                current_bs = min(self.genbatch, to_generate)

                preds, _ = self.model.sample_no_grad(current_bs, sl, multinomial=True,
                                                     temperature=self.temperature)
                sequences = [self.vocab.reconstruct(i) for i in preds]
                sequences = list(set(sequences))
                outputs += sequences
                outputs = list(set(outputs))
                to_generate = buffer_size - len(outputs)

        return outputs


    def sample_batch(self):
        env = self.environment
        bs = env.bs
        sl = env.sl
        outputs = self._sample_batch(bs, sl)
        env.batch_state[f'{self.name}_raw'] = outputs
        if outputs:
            env.batch_state.samples += outputs
            env.batch_state.sources += [self.name]*len(outputs)


    def _sample_batch(self, bs, sl):
        bs = int(bs * self.p_batch)
        sequences = []

        if bs > 0:

            preds, _ = self.model.sample_no_grad(bs, sl, z=None, multinomial=True,
                                                temperature=self.temperature)

            sequences = [self.vocab.reconstruct(i) for i in preds]

        return sequences

    def after_compute_reward(self):
        if self.p_batch>0. and self.track:
            batch_state = self.environment.batch_state
            log = self.environment.log
            rewards = batch_state.rewards.detach().cpu().numpy()
            sources = np.array(batch_state.sources)

            if self.name in sources:
                log.update_metric(f'{self.name}_rewards', rewards[sources==self.name].mean())
            else:
                log.update_metric(f'{self.name}_rewards', 0.)

    def after_sample(self):
        if self.p_batch > 0. and self.track:
            log = self.environment.log
            batch_state = self.environment.batch_state
            samples = batch_state.samples
            sources = np.array(batch_state.sources)==self.name

            samples = [samples[i] for i in range(len(samples)) if sources[i]]

            raw_samples = batch_state[f'{self.name}_raw']

            if len(samples)>0:
                diversity = len(set(raw_samples))/len(raw_samples)
                used = log.unique_samples
                novel = [i for i in samples if not i in used]
                percent_novel = len(novel)/len(samples)
            else:
                diversity = 0
                percent_novel = 0.

            valid = len(samples)/len(set(batch_state[f'{self.name}_raw']))

            log.update_metric(f'{self.name}_new', percent_novel)
            log.update_metric(f"{self.name}_diversity", diversity)
            log.update_metric(f"{self.name}_valid", valid)

# Cell

class PriorSampler(ModelSampler):
    '''
    PriorSampler - sampler class to draw samples
    from latent vectors generated by `prior`

    Inputs:

    - `vocab Vocab`: vocabulary for reconstructing samples

    - `model GenerativeModel`: model to sample from

    - `prior nn.Module`: prior to sample latent vectors from

    - `name str`: sampler name

    - `buffer_size int`: number of samples to generate during `build_buffer`

    - `p_batch float`: what percentage of batch
    samples should come from this sampler

    - `genbatch int`: generation batch size

    - `track bool`: if metrics from this sampler should be tracked

    - `track_losses bool`: if prior losses should be tracked
    (ignored if no prior loss is given)

    - `train bool`: if the prior should be trained (requires prior loss)

    - `train_all bool`: if the prior should be trained on all
    samples in a batch or just ones from this specific sampler

    - `prior_loss Optional`: loss function for prior. See
    `PriorLoss` for an example

    - `temperature float`: sampeling temperature
    '''
    def __init__(self, vocab, model, prior, name, buffer_size,
                 p_batch, genbatch, track=True, track_losses=True, train=True,
                 train_all=False, prior_loss=None,
                 temperature=1., opt_kwargs={}):
        super().__init__(vocab,
                         model,
                         name,
                         buffer_size,
                         p_batch,
                         genbatch,
                         track,
                         temperature)

        self.train = train
        self.track_losses = track_losses
        self.train_all = train_all
        self.set_prior(prior, opt_kwargs)
        self.prior_loss = prior_loss

    def setup(self):
        super().setup()
        if self.train and self.prior_loss is not None:
            self.environment.log.add_log(self.name+'_loss')
            if self.track_losses:
                self.environment.log.add_metric(self.name+'_loss')


    def set_prior(self, prior, opt_kwargs):
        self.prior = to_device(prior)
        if self.train:
            self.opt = optim.Adam(self.prior.parameters(), **opt_kwargs)

    def zero_grad(self):
        if self.train:
            self.opt.zero_grad()

    def step(self):
        if self.train:
            self.opt.step()

    def _build_buffer(self, sl):
        buffer_size = self.buffer_size
        outputs = []
        to_generate = buffer_size

        if buffer_size > 0:
            for batch in range(int(np.ceil(buffer_size/self.genbatch))):
                current_bs = min(self.genbatch, to_generate)

                z = self.prior.sample(current_bs)
                preds, _ = self.model.sample_no_grad(current_bs, sl, z=z, multinomial=True,
                                                     temperature=self.temperature)
                sequences = [self.vocab.reconstruct(i) for i in preds]
                sequences = list(set(sequences))
                outputs += sequences
                outputs = list(set(outputs))
                to_generate = buffer_size - len(outputs)

        return outputs

    def sample_batch(self):
        env = self.environment
        bs = env.bs
        sl = env.sl
        sequences, sample_latents = self._sample_batch(bs, sl)

        env.batch_state[f'{self.name}_raw'] = sequences
        env.batch_state.latent_data[self.name] = sample_latents

        if sequences:
            env.batch_state.samples += sequences
            env.batch_state.sources += [self.name]*len(sequences)

    def _sample_batch(self, bs, sl):
        bs = int(bs * self.p_batch)
        sequences = []
        sample_latents = []

        if bs > 0:

            z = self.prior.rsample(bs)

            preds, _ = self.model.sample_no_grad(bs, sl, z=z, multinomial=True,
                                                temperature=self.temperature)
            sequences = [self.vocab.reconstruct(i) for i in preds]

        return sequences, z

    def compute_loss(self):
        env = self.environment
        batch_state = env.batch_state

        if self.train and self.prior_loss is not None:
            if self.train_all:
                subset_name = None
            else:
                subset_name = self.name

            loss = self.prior_loss.from_batch_state(batch_state, subset_name)

            if self.track_losses:

                loss_d = loss.detach().cpu()

                if self.train_all:
                    mean_loss = loss_d.mean()
                else:
                    mean_loss = loss_d[loss_d.nonzero()[0]].mean()

                self.environment.log.update_metric(self.name+'_loss', mean_loss.numpy())

            self.environment.batch_state.loss += loss.mean()
            self.environment.batch_state[self.name+'_loss'] = loss.detach().cpu().numpy()


# Cell

class LatentSampler(ModelSampler):
    '''
    ModelSampler - sampler class to draw samples from a `GenerativeModel`

    Inputs:

    - `vocab Vocab`: vocabulary for reconstructing samples

    - `model GenerativeModel`: model to sample from

    - `latents torch.FloatTensor[n_latents, d_latents]`:
    tensor of latent vectors. `n_latents` can be any value

    - `name str`: sampler name

    - `buffer_size int`: number of samples to generate during `build_buffer`

    - `p_batch float`: what percentage of batch
    samples should come from this sampler

    - `genbatch int`: generation batch size

    - `track bool`: if metrics from this sampler should be tracked

    - `train bool`: if the latent vectors should be trained

    - `temperature float`: sampeling temperature
    '''
    def __init__(self, vocab, model, latents, name, buffer_size,
                 p_batch, genbatch, track=True, train=True,
                 temperature=1., opt_kwargs={}):
        super().__init__(vocab,
                         model,
                         name,
                         buffer_size,
                         p_batch,
                         genbatch,
                         track,
                         temperature)

        self.train = train
        self.set_latents(latents, opt_kwargs)

    def set_latents(self, latents, opt_kwargs):
        self.latents = to_device(latents)
        if self.train:
            self.latents = nn.Parameter(self.latents)
            self.opt = optim.Adam([self.latents], **opt_kwargs)
        else:
            self.latents._requires_grad(False)

    def zero_grad(self):
        if self.train:
            self.opt.zero_grad()

    def step(self):
        if self.train:
            self.opt.step()

    def _build_buffer(self, sl):
        return []

    def sample_batch(self):
        env = self.environment
        bs = env.bs
        sl = env.sl
        sequences, sample_latents = self._sample_batch(bs, sl)

        env.batch_state[f'{self.name}_raw'] = sequences
        env.batch_state.latent_data[self.name] = sample_latents

        if sequences:
            env.batch_state.samples += sequences
            env.batch_state.sources += [self.name]*len(sequences)

    def _sample_batch(self, bs, sl):
        bs = int(bs * self.p_batch)
        sequences = []
        sample_latents = []

        if bs > 0:

            latent_idxs = torch.randint(0, self.latents.shape[0]-1, (bs,))
            sample_latents = self.latents[latent_idxs]

            preds, _ = self.model.sample_no_grad(bs, sl, z=sample_latents, multinomial=True,
                                                temperature=self.temperature)
            sequences = [self.vocab.reconstruct(i) for i in preds]

        return sequences, sample_latents


# Cell

class ContrastiveSampler(Sampler):
    '''
    ContrastiveSampler - contrastive sampling wrapper. Uses
    `base_sampler` to generate source sequences. Then uses
    `output_model` to generate target sequences. Adds tuple
    pairs of `(source, target)` to batch/buffer

    Inputs:

    - `base_sampler Sampler`: base sampler to generate
    source samples

    - `vocab Vocab`: vocab for reconstruction

    - `dataset Base_Dataset`: dataset for tensorizing samples

    - `output_moodel GenerativeModel`: model to generate
    target samples

    - `bs int`: batch size for contrastive generation

    - `repeats int`: how many target samples to draw from
    each source sample
    '''
    def __init__(self, base_sampler, vocab, dataset, output_model, bs, repeats=1):
        super().__init__(base_sampler.name, base_sampler.buffer_size,
                         base_sampler.p_batch, base_sampler.track)

        self.base_sampler = base_sampler
        self.vocab = vocab
        self.dataset = dataset
        self.output_model = output_model
        self.bs = bs
        self.repeats = repeats

    def __call__(self, event_name):

        event = getattr(self, event_name, None)

        if event is not None:
            output = event()
        else:
            output = None

        if not (event_name in ['build_buffer', 'sample_batch']):
            _ = self.base_sampler(event_name)

        return output

    def setup(self):
        self.base_sampler.environment = self.environment

    def sample_outputs(self, sequences, sl):
        sequences = list(sequences)
        if self.repeats>1:
            sequences = sequences*self.repeats
        pairs = [(i,'') for i in sequences]
        batch_ds = self.dataset.new(pairs)

        if len(batch_ds)<self.bs:

            batch = batch_ds.collate_function([batch_ds[i] for i in range(len(batch_ds))])
            batch = to_device(batch)
            x,_ = batch
            z = self.output_model.x_to_latent(x)
            preds, _ = self.output_model.sample_no_grad(z.shape[0], sl, z=z)
            new_sequences = [self.vocab.reconstruct(i) for i in preds]

        else:
            batch_dl = batch_ds.dataloader(self.bs, shuffle=False)

            new_sequences = []

            for i, batch in enumerate(batch_dl):
                batch = to_device(batch)
                x,_ = batch
                z = self.output_model.x_to_latent(x)
                preds, _ = self.output_model.sample_no_grad(z.shape[0], sl, z=z)
                new_sequences += [self.vocab.reconstruct(i) for i in preds]

        outputs = [(sequences[i], new_sequences[i]) for i in range(len(sequences))]

        return outputs

    def _build_buffer(self):
        outputs = self.base_sampler._build_buffer()
        env = self.environment
        sl = env.sl
        if outputs:
            outputs = self.sample_outputs(outputs, sl)
        return outputs

    def _sample_batch(self):
        outputs = self.base_sampler._sample_batch()
        env = self.environment
        sl = env.sl
        if outputs:
            outputs = self.sample_outputs(outputs, sl)
            env.batch_state[f'{self.name}_raw_contrastive'] = outputs
        return outputs

# Cell

class LogSampler(Sampler):
    '''
    LogSampler - pulls samples from log
    based on `percentile`

    Inputs:

    - `sample_name str`: what column in `Log.df` to pull
    samples from

    - `lookup_name str`: what column in `Log.df` to use
    to find high scoring samples

    - `start_iter int`: iteration to start drawing from log

    - `percentile int`: value 1-100 percentile of
    log data to sample from

    - `buffer_size int`: number of samples to generate during `build_buffer`
    '''
    def __init__(self, sample_name, lookup_name, start_iter, percentile, buffer_size):
        super().__init__(sample_name+'_sample', buffer_size, p_batch=0.)
        self.start_iter = start_iter
        self.percentile = percentile
        self.sample_name = sample_name
        self.lookup_name = lookup_name

    def build_buffer(self):
        env = self.environment
        iterations = self.environment.log.iterations
        df = env.log.df

        outputs = self._build_buffer(iterations, df)
        if outputs:
            self.environment.buffer.add(outputs, self.name)

    def _build_buffer(self, iterations, df):
        outputs = []

        if iterations > self.start_iter:
            bs = self.buffer_size
            if bs > 0:

                subset = df[df[self.lookup_name]>np.percentile(df[self.lookup_name].values,
                                                               self.percentile)]
                outputs = list(subset.sample(n=min(bs, subset.shape[0]))[self.sample_name].values)

        return outputs

# Cell

class TokenSwapSampler(LogSampler):
    '''
    TokenSwapSampler - samples high scoring samples from
    `Log.df` and enumerates variants by swapping tokens.
    Note that token swapped samples are not guaranteed to be
    chemically valid. This sampler works best with SELFIES
    representation

    Inputs:

    - `sample_name str`: what column in `Log.df` to pull
    samples from

    - `lookup_name str`: what column in `Log.df` to use
    to find high scoring samples

    - `start_iter int`: iteration to start drawing from log

    - `percentile int`: value 1-100 percentile of
    log data to sample from

    - `buffer_size int`: number of samples to generate during `build_buffer`

    - `vocab Vocab`: vocab to numericalize samples

    - `swap_percent float`: percent of tokens to swap per sample
    '''
    # TODO: make version of this where tokens to swap are selected based on log probs
    def __init__(self, sample_name, lookup_name, start_iter, percentile, buffer_size,
                 vocab, swap_percent):
        super().__init__(sample_name, lookup_name, start_iter, percentile, buffer_size)
        self.name = sample_name+'_tokswap'
        self.vocab = vocab
        self.swap_percent = swap_percent

    def _build_buffer(self, iterations, log):
        samples = super()._build_buffer(iterations, log)

        new_samples = []

        for sample in samples:
            tokens = self.vocab.tokenize(sample)
            num_swaps = int(self.swap_percent*len(tokens))
            swap_idxs = np.random.choice(np.arange(len(tokens)), num_swaps, replace=False)
            new_tokens = np.random.choice(self.vocab.itos, num_swaps, replace=True)
            for idx, new_token in zip(swap_idxs, new_tokens):
                tokens[idx] = new_token

            sample = self.vocab.numericalize(tokens)
            sample = self.vocab.reconstruct(sample)
            new_samples.append(sample)

        return new_samples

# Cell

class LogEnumerator(LogSampler):
    '''
    LogEnumerator - pulls high scoring samples
    from `Log.df` and performs simple enumeration
    by adding one atom or one bond to the sample.
    Note that this proccess can create a large number
    samples and the value of `buffer_size` should
    accordingly be low (3-5). See `add_atom_combi` and
    `add_bond_combi` for more details on how the
    enumeration is done

    Inputs:

    - `sample_name str`: what column in `Log.df` to pull
    samples from

    - `lookup_name str`: what column in `Log.df` to use
    to find high scoring samples

    - `start_iter int`: iteration to start drawing from log

    - `percentile int`: value 1-100 percentile of
    log data to sample from

    - `buffer_size int`: number of samples to generate during `build_buffer`

    - `atom_types list`: list of allowed atom types to swap
    '''
    def __init__(self, sample_name, lookup_name, start_iter, percentile,
                 buffer_size, atom_types=None):
        super().__init__(sample_name, lookup_name, start_iter, percentile, buffer_size)

        self.name = sample_name+'_enum'
        if atom_types is None:
            atom_types = ['C', 'N', 'O', 'F', 'S', 'Br', 'Cl', -1]

        self.atom_types = atom_types

    def _build_buffer(self, iterations, df):
        samples = super()._build_buffer(iterations, df)

        new_samples = []

        for sample in samples:

            new_smiles = add_atom_combi(sample, self.atom_types) + add_bond_combi(sample)
            new_smiles = [i for i in new_smiles if i is not None]
            new_smiles = [i for i in new_smiles if not '.' in i]
            new_samples += new_smiles

        return new_samples

# Cell

class Timeout(Callback):
    '''
    Timeout - puts samples in "timeout" to prevent
    training on the same sample too frequently. Samples
    are only allowed to be trained on once every
    `timeout_length` batches

    Inputs:

    - `timeout_ength int`: number of batches to put
    molecule in timeout

    - `timeout_function Optional[Callable]`: preprocessing
    function for samples

    - `track bool`: if metrics from this callback should be tracked

    - `name str`: callback name
    '''
    def __init__(self, timeout_length, timeout_function=None,
                 track=True, name='timeout'):
        super().__init__(name=name)

        self.timeout_length = timeout_length
        self.timeout_dict = {}
        self.timeout_function = timeout_function
        self.track = track

    def setup(self):
        if self.track:
            log = self.environment.log
            log.add_metric(self.name)

    def filter_batch(self):
        batch_state = self.environment.batch_state
        samples = batch_state.samples

        valids = []

        for sample in samples:
            if self.timeout_function is not None:
                sample = self.timeout_function(sample)

            if sample in self.timeout_dict.keys():
                valids.append(False)

            else:
                valids.append(True)
                self.timeout_dict[sample] = self.timeout_length + 1

        self._filter_batch(valids)

        to_remove = []
        for k,v in self.timeout_dict.items():
            new_v = v-1
            self.timeout_dict[k] = new_v
            if new_v <= 0:
                to_remove.append(k)

        for key in to_remove:
            self.timeout_dict.pop(key)

        pct_valid = np.array(valids).mean()

        if self.track:
            log = self.environment.log
            log.update_metric(self.name, pct_valid)

# Cell

class MurckoTimeout(Timeout):
    '''
    MurckoTimeout - puts samples in "timeout" to prevent
    training on the same sample too frequently. Samples
    are only allowed to be trained on once every
    `timeout_length` batches. `MurckoTimeout`
    identifies samples by their Murcko scaffold

    Inputs:

    - `timeout_ength int`: number of batches to put
    molecule in timeout

    - `generic bool`: if True, Murcko scaffolds will be
    made generic (all carbon, single bonds) before evaluuation

    - `track bool`: if metrics from this callback should be tracked

    - `name str`: callback name
    '''
    def __init__(self, timeout_length, generic=False, track=True, name='murcko_timeout'):
        timeout_function = partial(murcko_scaffold, generic=generic)
        super().__init__(timeout_length, timeout_function, track, name)