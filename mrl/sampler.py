# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/19_samplers.ipynb (unless otherwise specified).

__all__ = ['Sampler', 'DatasetSampler', 'ModelSampler', 'LatentSampler', 'ContrastiveSampler', 'LogSampler',
           'TokenSwapSampler', 'LogEnumerator']

# Cell

from .imports import *
from .torch_imports import *
from .torch_core import *
from .callbacks import *
from .chem import *

# Cell

class Sampler(Callback):
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
    def __init__(self, data, buffer_size, name):
        super().__init__(name, buffer_size, 0.)
        self.data = data

    def _build_buffer(self):
        idxs = np.random.randint(0, len(self.data), self.buffer_size)
        samples = [self.data[i] for i in idxs]
        return samples

# Cell

class ModelSampler(Sampler):
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

            if len(samples)>0:
                diversity = len(set(samples))/len(samples)
                used = log.unique_samples
                novel = [i for i in samples if not i in used]
                percent_novel = len(novel)/len(samples)
            else:
                diversity = 0
                percent_novel = 0.

            valid = len(samples)/len(batch_state[f'{self.name}_raw'])

#             used = log.unique_samples
#             novel = [i for i in samples if not i in used]
#             percent_novel = len(novel)/len(samples)
            log.update_metric(f'{self.name}_new', percent_novel)
            log.update_metric(f"{self.name}_diversity", diversity)
            log.update_metric(f"{self.name}_valid", valid)

# Cell

class LatentSampler(ModelSampler):
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

class TokenSwapSampler(LogSampler):
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

            sample = self.vocab.join_tokens(tokens)
            sample = self.vocab.postprocess(sample)
            new_samples.append(sample)

        return new_samples

# Cell

class LogEnumerator(LogSampler):
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